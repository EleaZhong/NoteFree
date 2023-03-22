import os
import threading
import pyaudio
import wave
import requests
from queue import Queue
import logging
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TRANSCRIPTION_API_URL = "https://api.openai.com/v1/audio/transcriptions"
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}
TEXTTYPE = "university lecture" # "university lecture"


def set_up():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("notes", exist_ok=True)


def setup_recording_logger():
    recording_logger = logging.getLogger("recording")
    recording_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("logs/record.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    file_handler.setFormatter(formatter)
    recording_logger.addHandler(file_handler)
    return recording_logger


def setup_processing_logger():
    processing_logger = logging.getLogger("processing")
    processing_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("logs/process.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    file_handler.setFormatter(formatter)
    processing_logger.addHandler(file_handler)
    return processing_logger


set_up()
recording_logger = setup_recording_logger()
processing_logger = setup_processing_logger()


def record_audio(filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 120

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    recording_logger.info("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    recording_logger.info("Finished recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def transcribe_audio(filename):
    with open(filename, "rb") as f:
        response = requests.post(
            TRANSCRIPTION_API_URL,
            headers=headers,
            data={
                "model": "whisper-1",
            },
            files={"file": f}
        )
    return response.json()["text"]


def summarize_text(text, notes, summary, keywords):
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a professional, expert, and knowledgable assistant."},
            {
                "role": "user",
                "content": f"Transcription for a {TEXTTYPE}: <raw>{text}</raw>. The notes for this topic so far: <notes>{notes}</notes>. The summary so far: <summary>{summary}</summary>. The keywords for this topic so far: <keywords>{keywords}</keywords>. Return new notes in succinct bullet points with the format like <notes>- text\n- text\n...</notes>, not neccesary to write a whole sentence. The new notes in total should be shorter than the length of the current transcription; notes should not have more information than is given in the context. Write a comprehensive summary that summarizes the topic, keep in mind both the past and current notes, like <summary>...</summary>, in addition to a literal summary, provide some analysis, synthesis, open-ended questions, and ideas in the summary, feel free to write multiple paragraphs. Provide a few keywords and key phrases separated by commas, like <keywords>...</keywords>. Keywords and summary should be for overall content, not just the current transcript. Title the topic in 3-5 words, like <topic>...</topic>. Use your knowledge to correct the transcript only if there is any factual errors." 
            }
        ]
    }
    processing_logger.info("Data: " + str(data))
    response = requests.post(
        CHAT_API_URL,
        headers=headers,
        json=data
    )
    return response.json()['choices'][0]['message']['content']


def summarize_notes(notes):
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a professional, expert, and knowledgable assistant."},
            {
                "role": "user",
                "content": f"Summarize these notes from a {TEXTTYPE}: \n{notes}\nReturn 3 lines of notes in succinct bullet points with the format like <notes>- text\n- text\n...</notes>, not neccesary to write a whole sentence." 
            }
        ]
    }
    processing_logger.info("Summarize Data: " + str(data))
    response = requests.post(
        CHAT_API_URL,
        headers=headers,
        json=data
    )
    return response.json()['choices'][0]['message']['content']


def print_and_write(notes, summary, keywords, topicname, topic_counter=0):
    print("################")
    print(f"Topic {topicname} Notes:")
    print(notes)
    print(f"Topic {topicname} Summary:")
    print(summary)
    print(f"Topic {topicname} Keywords:")
    print(keywords)
    print("################")
    with open(f"notes/{topic_counter:02d}-{topicname}.md", "w") as f:
        f.write("# Summary\n\n")
        f.write(summary + "\n\n")
        f.write("# Notes\n\n")
        f.write(notes + "\n\n")
        f.write("# Keywords\n\n")
        f.write(keywords)



def retrieve_text_from_tag(text, tag):
    if f"<{tag}>" in text and f"</{tag}>" in text:
        return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()
    else:
        return ""


def process_audio(queue):
    notes = ""
    summary = ""
    keywords = ""
    topic = ""
    topic_counter = 0
    while True:
        try:
            recording_number = queue.get()
        except KeyboardInterrupt:
            print_and_write(notes, summary, keywords, topic, topic_counter)
            exit()
        filename = f"recordings/recording-{recording_number:04d}.wav"
        transcript = transcribe_audio(filename)
        assistant_content = summarize_text(transcript, notes, summary, keywords)

        new_notes = retrieve_text_from_tag(assistant_content, "notes")
        notes = notes + "\n" + new_notes if new_notes else notes
        summary = retrieve_text_from_tag(assistant_content, "summary")
        keywords = retrieve_text_from_tag(assistant_content, "keywords")
        topic = retrieve_text_from_tag(assistant_content, "topic")

        processing_logger.info("Transcript: " + transcript)
        processing_logger.info("Notes: " + notes)
        processing_logger.info("New notes: " + new_notes)
        processing_logger.info("Summary: " + summary)
        processing_logger.info("Keywords: " + keywords)
        processing_logger.info("Topic: " + topic)
        processing_logger.info("Assistant content: " + assistant_content)
        processing_logger.info("################")

        print(new_notes)

        if len(notes)>=1000:
            processing_logger.info("Notes too long, splitting")
            print_and_write(notes, summary, keywords, topic, topic_counter)
            notes = summarize_notes(notes)
            processing_logger.info("Summarize Notes:" + notes)
            keywords = ""
            topic_counter += 1    
        


def continuous_recording(queue):
    recording_number = 0
    while True:
        filename = f"recordings/recording-{recording_number:04d}.wav"
        record_audio(filename)
        recording_logger.info("Filename: " + str(filename))
        queue.put(recording_number)
        recording_number += 1


if __name__ == "__main__":
    recording_queue = Queue()

    recording_thread = threading.Thread(
        target=continuous_recording, args=(recording_queue,))
    recording_thread.start()

    processing_thread = threading.Thread(
        target=process_audio, args=(recording_queue,))
    processing_thread.start()
    