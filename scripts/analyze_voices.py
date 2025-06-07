import json

with open("all_elevenlabs_voices.json", "r") as f:
    voices = json.load(f)

print("🎙️ VOZES DISPONÍVEIS NO ELEVENLABS:")
print("=" * 80)

male_voices = []
female_voices = []
neutral_voices = []

for voice in voices:
    gender = voice["labels"].get("gender", "unknown")
    name = voice["name"]
    voice_id = voice["voice_id"]
    accent = voice["labels"].get("accent", "unknown")
    age = voice["labels"].get("age", "unknown")

    print(f"{name:12} | {gender:7} | {accent:10} | {age:12} | {voice_id}")

    if gender == "male":
        male_voices.append(voice)
    elif gender == "female":
        female_voices.append(voice)
    else:
        neutral_voices.append(voice)

print(f"\n📊 RESUMO:")
print(f"👨 Masculinas: {len(male_voices)}")
print(f"👩 Femininas: {len(female_voices)}")
print(f"⚡ Neutras: {len(neutral_voices)}")
