import argparse
import librosa
import torch
from omegaconf import OmegaConf
from transformers import LlamaTokenizer, AutoTokenizer, HubertForCTC

from model.audio_encoder import AudioEncoder
from model.audio_llama import AudioLlamaForCausalLM
from utils import merge_prompt_tokens, PROMPT_PREFIX, PROMPT_SUFFIX
import pandas as pd
from tqdm import tqdm


class LLMSpeechTextInference():
    def __init__(self, config, audio_encoder_checkpoint, devices):
        self.config = config
        self.devices = devices

        # Audio encoder.
        checkpoint = torch.load(audio_encoder_checkpoint, map_location="cpu")
        self.audio_encoder = AudioEncoder(self.config)
        self.audio_encoder.load_state_dict(checkpoint)
        self.audio_encoder.eval().to(self.devices[0])
        print("Loaded audio encoder.\n")

        # LLM tokenizer.
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            "GeneZC/MiniChat-2-3B",
            use_fast=False,
        )

        # Load and freeze LLM model weights.
        self.llm = AudioLlamaForCausalLM.from_pretrained(
            "GeneZC/MiniChat-2-3B",
            use_cache=True,
            torch_dtype=torch.float16,
        ).eval()
        self.llm.to(self.devices[1])
        print("Loaded LLM.\n")

        # Load HuBERT ASR model for getting CTC offsets.
        if (self.audio_encoder.downsample_method == "ctc_pool"):
            self.hubert_tokenizer = AutoTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
            self.hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(devices[0])
            self.hubert.to(self.devices[0])
            print("Loaded HuBERT.\n")
        
        print("done")

    def perform_hubert_asr(self, audio):
        # Feed audio through model to get greedily predicted transcription IDs.
        logits = self.hubert(audio).logits[0]
        pred_ids = torch.argmax(logits, axis=-1)

        # Decode transcription IDs to get text transcript.
        # NOTE: Always converts to lower case.
        transcript = self.hubert_tokenizer.decode(pred_ids).lower()
        return transcript

    def get_ctc_pool_ranges(self, audio, pool_range=4):
        # Feed audio through model to get greedily predicted transcription IDs.
        logits = self.hubert(audio).logits[0]
        pred_ids = torch.argmax(logits, axis=-1)

        # Perform decoding to get CTC offsets for each predicted word.
        outputs = self.hubert_tokenizer.decode(pred_ids, output_word_offsets=True)
        word_offsets = outputs.word_offsets
        ctc_word_offsets = [
            (word['start_offset'], word['end_offset']) for word in word_offsets
        ]

        # Add offset ranges for silence in between words. The first element of
        # each tuple is a flag denoting whether the offset corresponds to
        # a word (1) or silence (0).
        all_word_offsets = [(0, 0, ctc_word_offsets[0][0])]
        for i in range(len(ctc_word_offsets)-1):
            all_word_offsets.append((1, ctc_word_offsets[i][0], ctc_word_offsets[i][1]))
            all_word_offsets.append((0, ctc_word_offsets[i][1], ctc_word_offsets[i+1][0]))
        all_word_offsets.append((1, ctc_word_offsets[-1][0], ctc_word_offsets[-1][1]))
        all_word_offsets.append(
            (0, ctc_word_offsets[-1][1], ctc_word_offsets[-1][1] + (pool_range * 2))
        )

        # Aggregate the offsets into pooling ranges for the audio encoder.
        ctc_pool_ranges = []
        for is_word, start_offset, end_offset in all_word_offsets:
            if is_word == 1:
                startpoint = start_offset
                endpoint = start_offset + pool_range
                while startpoint < end_offset:
                    ctc_pool_ranges.append((startpoint, endpoint))
                    startpoint += pool_range
                    endpoint += pool_range
            else:
                ctc_pool_ranges.append((start_offset, end_offset))

        return ctc_pool_ranges

    def generate_llm_response(self, inputs_embeds, max_new_tokens=256):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # NOTE: Using greedy decoding for generation (no sampling).
                # Uncomment the lines below to change this.
                generate_ids = self.llm.generate(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    # do_sample=True,
                    # temperature=0.7,
                    max_new_tokens=max_new_tokens,
                )

        response_text = self.llm_tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return response_text

    def generate_text_response(self, input_text, max_new_tokens=256):
        # Create full prompt for instruction-tuned LLM.
        # full_text_prompt = f"{PROMPT_PREFIX} {input_text}{PROMPT_SUFFIX} "
        full_text_prompt = input_text

        with torch.no_grad():
            # Tokenize and get embeddings for the full text prompt.
            prompt_input_ids = self.llm_tokenizer(
                full_text_prompt, return_tensors='pt'
            ).input_ids.to(self.devices[1])
            prompt_embeds = self.llm.model.embed_tokens(prompt_input_ids)

            # Generate the LLM response.
            llm_response = self.generate_llm_response(
                inputs_embeds=prompt_embeds,
                max_new_tokens=max_new_tokens,
            )[0]

        return llm_response

    def generate_asr_cascade_response(self, audio, additional_text_prompt="", max_new_tokens=256):
        with torch.no_grad():
            # Perform ASR using HuBERT.
            audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.devices)
            asr_transcript = self.perform_hubert_asr(audio_tensor)

            # Combine the transcript with any additional text prompt.
            # NOTE: Assumes that the text prompt always comes before the
            # transcribed text.
            full_text = additional_text_prompt + asr_transcript
            llm_response = self.generate_text_response(full_text, max_new_tokens)

        return llm_response

    def generate_audio_response(self, audio, additional_text_prompt="", max_new_tokens=256):
        with torch.no_grad():
            audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.devices[0])

            if self.audio_encoder.downsample_method == "ctc_pool":
                # Get the CTC pooling ranges for the audio.
                ctc_pool_ranges = self.get_ctc_pool_ranges(audio_tensor)

                # Get embeddings from the audio encoder.
                audio_embeds = self.audio_encoder(audio_tensor, [ctc_pool_ranges])
            else:
                audio_embeds = self.audio_encoder(audio_tensor, ctc_pool_ranges=None)
            audio_embeds = audio_embeds.to(self.devices[1]) 

            # Combine the audio embeddings with any additional text prompt.
            # NOTE: Currently assumes that the text prompt always comes before
            # the audio. You can change how the embeddings are concatenated to
            # switch up the order or interleave text and audio prompts.
            if len(additional_text_prompt) > 0:
                # Take elements [1:] to remove start of sentence token.
                additional_text_input_ids = self.llm_tokenizer(
                    additional_text_prompt, return_tensors='pt'
                ).input_ids[:, 1:].to(self.devices[1])

                # Get embeddings corresponding to additional text prompt and
                # concatenate with audio embeddings.
                text_embeds = self.llm.model.embed_tokens(additional_text_input_ids)
                combined_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
            else:
                # Otherwise, just use the audio embeddings.
                combined_embeds = audio_embeds

            # Get the full embedding sequence and generate the LLM response
            prompt_emb_sequence = merge_prompt_tokens(
                inputs_embeds=combined_embeds,
                tokenizer=self.llm_tokenizer,
                embed_tokens=self.llm.model.embed_tokens,
                device=self.devices[1],
            )
            llm_response = self.generate_llm_response(prompt_emb_sequence, max_new_tokens)[0]

        return llm_response

def multiple_inference(config_path:str,gpu_idx:int,audio_encoder_checkpoint_path:str,audio_dir:str,audio_ids:list[str],output_file_path:str,user_prompt:str="Summarize the following article in 3 sentences or less"):
    """
    Perform multiple inferences on audio files and generate summaries using LLMSpeechTextInference.

    Args:
        config_path (str): Path to the configuration file.
        gpu_idx (int): Index of the GPU to use for running models.
        audio_encoder_checkpoint_path (str): Path to the audio encoder checkpoint.
        audio_dir (str): Directory containing the audio files.
        audio_ids (list[str]): List of audio file IDs.
        output_file_path (str): Path to the output file where the summaries will be saved.

    Returns:
        None
    """
    
    # Select device for running models.
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")

    # Set up inferencer.
    config = OmegaConf.load(config_path)
    llm_inferencer = LLMSpeechTextInference(
        config=config,
        audio_encoder_checkpoint=audio_encoder_checkpoint_path,
        devices=device,
    )
    summaries = []
    for id in tqdm(audio_ids):
        # Load audio file.
        audio, sr = librosa.load(f"{audio_dir}/{id}.wav", sr=16000)

        # Generate LLM response.
        # NOTE: Generating the response in this way sometimes leads to the LLM repeating a
        # chunk of text over and over. You can manually get around this by cropping the
        # generated output.
        llm_response = llm_inferencer.generate_audio_response(
            audio,
            additional_text_prompt=user_prompt,
            max_new_tokens=512,
        )
        llm_response = llm_response.split("\n")
        summaries.append((id,llm_response[0]))
    
    df = pd.DataFrame(summaries,columns=["id","summary"])
    df.to_csv(output_file_path,index = False)

def load_model(config_path, gpu_idx, audio_encoder_checkpoint_path)->LLMSpeechTextInference:
    """
    Loads the LLMSpeechTextInference model with the specified configuration, GPU index, and audio encoder checkpoint path.

    Args:
        config_path (str): The path to the configuration file.
        gpu_idx (int): The index of the GPU to use for inference.
        audio_encoder_checkpoint_path (str): The path to the audio encoder checkpoint.

    Returns:
        LLMSpeechTextInference: The loaded LLMSpeechTextInference model.
    """
    device1 = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    device2 = torch.device(f"cuda:{gpu_idx+1}" if torch.cuda.is_available() else "cpu")

    # Set up inferencer.
    config = OmegaConf.load(config_path)
    llm_inferencer = LLMSpeechTextInference(
        config=config,
        audio_encoder_checkpoint=audio_encoder_checkpoint_path,
        devices=[device1,device2],
    )

    return llm_inferencer

def user_inference(llm_inferencer:LLMSpeechTextInference,audio_dir:str,user_prompt:str=""):
    """
    Perform inference using LLMSpeechTextInference model.

    Args:
        llm_inferencer (LLMSpeechTextInference): The LLMSpeechTextInference model.
        audio_dir (str): The directory path of the audio file.
        user_prompt (str, optional): Additional text prompt for the model. Defaults to "".

    Returns:
        None
    """
    if(llm_inferencer is None):
        print("Please load the model first")
        return
    # Load audio file.
    audio, sr = librosa.load(f"{audio_dir}", sr=16000)

    # Generate LLM response.
    # NOTE: Generating the response in this way sometimes leads to the LLM repeating a
    # chunk of text over and over. You can manually get around this by cropping the
    # generated output.
    
    llm_response = llm_inferencer.generate_audio_response(
        audio,
        additional_text_prompt=user_prompt,
        max_new_tokens=512,
    )
    print(llm_response)
    torch.cuda.empty_cache()


    

if __name__ == '__main__':
    """
    Example use case for running generate_audio_response.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="yaml file for configuration")
    parser.add_argument('-g', '--gpu_idx', type=int, default=0,
                        help="index of home GPU device")
    parser.add_argument('-p', '--audio_encoder_checkpoint', type=str,
                        help="path to audio encoder checkpoint")
    parser.add_argument('-a', '--audio_file', type=str, required=True,
                        help="audio file containing speech utterance to be used in prompt")
    args = parser.parse_args()

    # Select device for running models.
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")

    # Set up inferencer.
    config = OmegaConf.load(args.config)
    llm_inferencer = LLMSpeechTextInference(
        config=config,
        audio_encoder_checkpoint=args.audio_encoder_checkpoint,
        devices=device,
    )

    # Load audio file.
    audio, sr = librosa.load(args.audio_file, sr=16000)

    # Generate LLM response.
    # NOTE: Generating the response in this way sometimes leads to the LLM repeating a
    # chunk of text over and over. You can manually get around this by cropping the
    # generated output.
    llm_response = llm_inferencer.generate_audio_response(
        audio,
        additional_text_prompt="Summarize the following article in 3 sentences or less",
        max_new_tokens=512,
    )

    print("LLM Response:\n")
    # split llm response to paragraphs and print the first one
    llm_response = llm_response.split("\n")
    print(llm_response[0])
    

