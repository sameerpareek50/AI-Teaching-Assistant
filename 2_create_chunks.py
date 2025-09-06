# Step 2 : convert mp3 into chunks(texts) using whisper large-v2 model

# i used this script to create chunks from audio files using whisper and save them as json files with metadata

import whisper
import json
import os

model = whisper.load_model("large-v2")

audios = os.listdir("audios")

for audio in audios: 
    if("_" in audio):  # yeh esliye likhi kyuki initially ek sample mp3 pr sab test kr rha tha jiske naam m _ nhi tha to usme error aa rha tha
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4] # remove .mp3 extension
        print(number, title)
        result = model.transcribe(audio = f"audios/{audio}", 
        # result = model.transcribe(audio = f"audios/sample.mp3", 
                              language="hi",
                              task="translate",
                              word_timestamps=False )
        
        chunks = [] # list of chunks with metadata

        for segment in result["segments"]: # har segment k liye ek dictionary banao jisme start, end, text, number, title sab ho
            # segment wasn an array jisme pura bada sa data tha with various info but we chose only important info for our use case
            chunks.append({"number": number, "title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
        
        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
        # yeh mene likha bcoz m chahta tha pura text of that video bhi sath m rakhe rakhna for reference toh chunks m toh imp info aa gyi aur text m saara text aa gya 

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata,f)
            # esliye mene yahan pr neeche chunks_with_metadata likha kyuki usme pura text bhi tha aur chunks bhi the with metadata
            # agar mene sirf chunks likha hota toh usme pura text nhi hota 