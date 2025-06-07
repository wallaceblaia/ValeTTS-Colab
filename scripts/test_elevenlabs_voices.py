#!/usr/bin/env python3
"""
Script para testar API ElevenLabs e obter metadados das vozes.
"""

import json
import os

from elevenlabs import ElevenLabs, VoiceSettings


def test_elevenlabs_api():
    """Testa a API ElevenLabs e obtém metadados das vozes."""

    # IDs das vozes fornecidas pelo usuário
    VOICE_IDS = [
        "EkK5I93UQWFDigLMpZcX",
        "RILOU7YmBhvwJGDGjNmP",
        "Bj9UqZbhQsanLzgalpEG",
        "exsUS4vynmxd379XN4yO",
        "SOYHLrjzK2X1ezoPC6cr",
        "CmoLJ5IFIu3FQ1FZAYIK",
        "0TfZ4rvne3QI7UjDxVkM",
        "hgvN1QdArfyEaEu5ZvPH",
        "n9RJRb8CoeRtSYSpKZRH",
        "prNO3mjGt58JtxmVSe74",
        "8PwhAQYisogJcn6egCE6",
        "AiVXo6AkAsMPEX0qNgmP",
        "jwjWpCFQUCpnHneBySsF",
        "qkskiKFtn5qTrNTRzb6M",
        "V1sMnmZsNBJuEOA7aVqB",
        "rTOopItG6FIkKMIVxsl5",
        "pIsMvEB8LP1GR5k3OcQj",
        "ug7mg45jVbzgYHpQBrw5",
    ]

    # Verificar se a chave API está configurada
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ ELEVENLABS_API_KEY não encontrada no ambiente!")
        return

    print(f"🔑 API Key configurada: {api_key[:10]}...")

    # Inicializar cliente ElevenLabs
    try:
        client = ElevenLabs(api_key=api_key)
        print("✅ Cliente ElevenLabs inicializado")
    except Exception as e:
        print(f"❌ Erro ao inicializar cliente: {e}")
        return

    # Obter todas as vozes disponíveis
    print("\n🎙️ Obtendo todas as vozes disponíveis...")
    try:
        all_voices = client.voices.get_all()
        print(f"✅ Total de vozes disponíveis: {len(all_voices.voices)}")

        # Salvar todas as vozes para referência
        all_voices_data = []
        for voice in all_voices.voices:
            voice_data = {
                "voice_id": voice.voice_id,
                "name": voice.name,
                "category": getattr(voice, "category", "unknown"),
                "labels": getattr(voice, "labels", {}),
                "description": getattr(voice, "description", ""),
                "preview_url": getattr(voice, "preview_url", ""),
                "available_for_tiers": getattr(voice, "available_for_tiers", []),
                "settings": getattr(voice, "settings", None),
            }
            all_voices_data.append(voice_data)

        # Salvar arquivo JSON com todas as vozes
        with open("all_elevenlabs_voices.json", "w", encoding="utf-8") as f:
            json.dump(all_voices_data, f, indent=2, ensure_ascii=False)

        print("📄 Todas as vozes salvas em: all_elevenlabs_voices.json")

    except Exception as e:
        print(f"❌ Erro ao obter vozes: {e}")
        return

    # Filtrar e analisar apenas as vozes especificadas
    print(f"\n🎯 Analisando {len(VOICE_IDS)} vozes especificadas...")

    target_voices = []
    found_ids = []

    for voice in all_voices.voices:
        if voice.voice_id in VOICE_IDS:
            found_ids.append(voice.voice_id)

            voice_info = {
                "voice_id": voice.voice_id,
                "name": voice.name,
                "category": getattr(voice, "category", "unknown"),
                "labels": dict(getattr(voice, "labels", {})),
                "description": getattr(voice, "description", ""),
                "preview_url": getattr(voice, "preview_url", ""),
                "available_for_tiers": getattr(voice, "available_for_tiers", []),
                "settings": {
                    "stability": (
                        getattr(voice.settings, "stability", 0.75)
                        if voice.settings
                        else 0.75
                    ),
                    "similarity_boost": (
                        getattr(voice.settings, "similarity_boost", 0.75)
                        if voice.settings
                        else 0.75
                    ),
                    "style": (
                        getattr(voice.settings, "style", 0.0) if voice.settings else 0.0
                    ),
                    "use_speaker_boost": (
                        getattr(voice.settings, "use_speaker_boost", True)
                        if voice.settings
                        else True
                    ),
                },
            }

            target_voices.append(voice_info)

    # Relatório das vozes encontradas
    print(f"✅ Encontradas {len(found_ids)}/{len(VOICE_IDS)} vozes especificadas")

    # IDs não encontrados
    missing_ids = set(VOICE_IDS) - set(found_ids)
    if missing_ids:
        print(f"❌ IDs não encontrados: {list(missing_ids)}")

    # Mostrar detalhes das vozes encontradas
    print(f"\n📋 DETALHES DAS VOZES ESPECIFICADAS:")
    print("=" * 80)

    # Agrupar por gênero e características
    male_voices = []
    female_voices = []
    other_voices = []

    for voice in target_voices:
        print(f"\n🎤 {voice['name']} (ID: {voice['voice_id']})")
        print(f"   📂 Categoria: {voice['category']}")
        print(f"   🏷️  Labels: {voice['labels']}")
        print(f"   📝 Descrição: {voice['description'][:100]}...")
        print(f"   🎛️  Configurações:")
        print(f"      - Estabilidade: {voice['settings']['stability']}")
        print(f"      - Similaridade: {voice['settings']['similarity_boost']}")
        print(f"      - Estilo: {voice['settings']['style']}")
        print(f"      - Speaker Boost: {voice['settings']['use_speaker_boost']}")

        # Tentar inferir gênero pelos labels ou descrição
        labels_text = str(voice["labels"]).lower()
        desc_text = voice["description"].lower()
        name_text = voice["name"].lower()

        if any(
            word in labels_text + desc_text + name_text
            for word in ["male", "man", "masculine"]
        ):
            male_voices.append(voice)
            print(f"   👨 Gênero inferido: MASCULINO")
        elif any(
            word in labels_text + desc_text + name_text
            for word in ["female", "woman", "feminine"]
        ):
            female_voices.append(voice)
            print(f"   👩 Gênero inferido: FEMININO")
        else:
            other_voices.append(voice)
            print(f"   ❓ Gênero inferido: INDETERMINADO")

    # Salvar dados das vozes especificadas
    with open("target_elevenlabs_voices.json", "w", encoding="utf-8") as f:
        json.dump(target_voices, f, indent=2, ensure_ascii=False)

    print(
        f"\n💾 Dados das vozes especificadas salvos em: target_elevenlabs_voices.json"
    )

    # Resumo final
    print(f"\n📊 RESUMO FINAL:")
    print(f"   👨 Vozes masculinas: {len(male_voices)}")
    print(f"   👩 Vozes femininas: {len(female_voices)}")
    print(f"   ❓ Vozes indeterminadas: {len(other_voices)}")
    print(f"   🎯 Total encontradas: {len(target_voices)}")

    # Testar síntese com uma voz
    if target_voices:
        print(f"\n🧪 TESTE DE SÍNTESE:")
        test_voice = target_voices[0]
        test_text = "Olá, este é um teste de síntese de voz usando ElevenLabs."

        try:
            print(f"🎙️ Testando voz: {test_voice['name']}")

            audio = client.generate(
                text=test_text,
                voice=test_voice["voice_id"],
                voice_settings=VoiceSettings(
                    stability=test_voice["settings"]["stability"],
                    similarity_boost=test_voice["settings"]["similarity_boost"],
                    style=test_voice["settings"]["style"],
                    use_speaker_boost=test_voice["settings"]["use_speaker_boost"],
                ),
            )

            # Salvar áudio de teste
            with open("test_elevenlabs_sample.mp3", "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            print("✅ Teste de síntese bem-sucedido!")
            print("🔊 Arquivo salvo: test_elevenlabs_sample.mp3")

        except Exception as e:
            print(f"❌ Erro no teste de síntese: {e}")


if __name__ == "__main__":
    test_elevenlabs_api()
