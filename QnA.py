#############
#
#    Question and Answering Bot for youtube videos
#
##############

'''
    Step 1. Extract the Youtube video 
'''
from pytube import YouTube
yt = YouTube('https://www.youtube.com/watch?v=MVYrJJNdrEg')

audio = yt.streams.filter(only_audio=True)
for file in audio:
   #xwprint(f"file is {file} \n")
    if file.mime_type == "audio/mp4":
        stream = yt.streams.get_by_itag(file.itag)
        print(f"file - {file}")
        stream.download()


'''
    Step 2. Convert the mp4 to mp3 
'''
from moviepy.editor import *
def MP4ToMP3(mp4, mp3):
    FILETOCONVERT = AudioFileClip(mp4)
    FILETOCONVERT.write_audiofile(mp3)
    FILETOCONVERT.close()

VIDEO_FILE_PATH = "podcast.mp4"
AUDIO_FILE_PATH = "new_podcast.mp3"
MP4ToMP3(VIDEO_FILE_PATH, AUDIO_FILE_PATH)

'''
    Step 3. Slice the audio clip into 1 mintue clips to a smaller sizes 
            for whisper to process 
'''
from pydub import AudioSegment
song = AudioSegment.from_mp3("../audio_files/new_podcast.mp3")
one_minute = 1 * 60 * 1000

start = 0
end = one_minute
minute = 1
while end < len(song):

    end = min(end, len(song))
    new_chunk = song[start:end]
    new_chunk.export(f"{minute}_minute.mp3", format="mp3")
    minute += 1
    start = end
    end += one_minute

'''
    Step 4. Transcribe and store the audio files
'''
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import whisper
import torch


import pinecone
pinecone.init(      
	api_key='dc653e06-ac6f-4f95-8c3c-0b412973b2fe',      
	environment='gcp-starter'
)      
client = pinecone.Index('podcast')
index_name = 'podcast'


device = "cuda" if torch.cuda.is_available() else "cpu"
modelE = SentenceTransformer('all-mpnet-base-v2')
model = whisper.load_model('large').to(device)

for i in range(3): #total number of audio files 
    a_f = "./video_slices/" + f"{i+1}" + "_minute.mp3"
    unique_id = f"processing {i+1}_minute.mp3"
    results = model.transcribe(a_f)
    embedd = modelE.encode(results['text']).tolist()
    client.upsert(
            vectors=[
                {'id': unique_id,
                'values': embedd,
                'metadata': {'text': results['text']}
                }
            ]
        )

