from deepface import DeepFace

# Provide the image path (replace with your image)
image_path = 'ospan.jpg'

# Perform facial analysis
analysis = DeepFace.analyze(image_path, actions=['age', 'gender', 'emotion', 'race'])

# Extracting results
age = analysis[0]['age']
gender = analysis[0]['dominant_gender']
dominant_emotion = analysis[0]['dominant_emotion']
dominant_race = analysis[0]['dominant_race']

# Print results in a list format
print("\n🔍 Facial Analysis Result:")
print(f"1️⃣ Age: {age}")
print(f"2️⃣ Gender: {gender}")
print(f"3️⃣ Emotion: {dominant_emotion}")
print(f"4️⃣ Race: {dominant_race}")

# Print detailed emotion and race breakdown
print("\n📌 Emotion Breakdown:")
for emotion, value in analysis[0]['emotion'].items():
    print(f"   - {emotion.capitalize()}: {value:.2f}%")

print("\n📌 Race Breakdown:")
for race, value in analysis[0]['race'].items():
    print(f"   - {race.capitalize()}: {value:.2f}%")

