#!/usr/bin/env python3
"""
Script para testar as vozes americanas do ElevenLabs.
"""

import asyncio
import sys
from pathlib import Path

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

from valetts.data.generation.elevenlabs_interface import ElevenLabsInterface


async def test_american_voices():
    """Testa todas as vozes americanas disponíveis."""

    try:
        # Inicializar interface
        print("🎙️ Inicializando ElevenLabs Interface...")
        interface = ElevenLabsInterface(output_dir="test_american_voices")

        # Obter vozes disponíveis
        voices = await interface.get_available_voices()
        print(f"✅ {len(voices)} vozes americanas carregadas")

        # Mostrar resumo das vozes
        print("\n📋 VOZES AMERICANAS DISPONÍVEIS:")
        print("=" * 80)

        male_voices = [v for v in voices if v["gender"] == "male"]
        female_voices = [v for v in voices if v["gender"] == "female"]
        neutral_voices = [v for v in voices if v["gender"] == "neutral"]

        print(f"\n👨 MASCULINAS ({len(male_voices)}):")
        for voice in male_voices:
            print(f"   🎤 {voice['name']:10} | {voice['age']:12} | {voice['voice_id']}")

        print(f"\n👩 FEMININAS ({len(female_voices)}):")
        for voice in female_voices:
            print(f"   🎤 {voice['name']:10} | {voice['age']:12} | {voice['voice_id']}")

        print(f"\n⚡ NEUTRAS ({len(neutral_voices)}):")
        for voice in neutral_voices:
            print(f"   🎤 {voice['name']:10} | {voice['age']:12} | {voice['voice_id']}")

        # Textos de teste
        test_texts = [
            "Olá, meu nome é {name} e eu sou uma voz americana do ElevenLabs.",
            "Este é um teste de síntese de voz em português brasileiro.",
            "A tecnologia de inteligência artificial está evoluindo rapidamente.",
        ]

        print(f"\n🧪 TESTANDO SÍNTESE COM {len(test_texts)} TEXTOS...")

        # Testar uma voz de cada gênero
        test_voices = []
        if male_voices:
            test_voices.append(("masculina", male_voices[0]))
        if female_voices:
            test_voices.append(("feminina", female_voices[0]))
        if neutral_voices:
            test_voices.append(("neutra", neutral_voices[0]))

        for gender_desc, voice_info in test_voices:
            print(f"\n🎙️ Testando voz {gender_desc}: {voice_info['name']}")

            for i, text_template in enumerate(test_texts):
                text = text_template.format(name=voice_info["name"])
                output_path = (
                    f"test_american_voices/{voice_info['name'].lower()}_{i+1}.mp3"
                )

                print(f"   📝 Texto {i+1}: {text[:60]}...")

                success = await interface.synthesize_text(
                    text=text,
                    voice_id=voice_info["voice_id"],
                    output_path=output_path,
                    style="natural",
                )

                if success:
                    print(f"   ✅ Áudio gerado: {output_path}")
                else:
                    print(f"   ❌ Falha na síntese")

        # Mostrar estatísticas finais
        stats = interface.get_statistics()
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"   🎯 Total gerado: {stats['total_generated']}")
        print(f"   ✅ Sucessos: {stats['successful']}")
        print(f"   ❌ Falhas: {stats['failed']}")
        print(f"   💰 Custo estimado: {stats['estimated_cost_usd']}")
        print(f"   📝 Total de caracteres: {stats['total_characters']}")

        if stats["successful"] > 0:
            print(f"\n🎉 Teste concluído com sucesso!")
            print(f"🔊 Arquivos salvos em: test_american_voices/")

    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback

        traceback.print_exc()


async def test_voice_selection():
    """Testa sistema de seleção de vozes."""

    print("\n🔍 TESTANDO SELEÇÃO DE VOZES...")

    interface = ElevenLabsInterface()

    # Testar diferentes critérios
    test_criteria = [
        {"gender": "male", "age": "young"},
        {"gender": "male", "age": "middle_aged"},
        {"gender": "male", "age": "old"},
        {"gender": "female", "age": "young"},
        {"gender": "female", "age": "middle_aged"},
        {"gender": "neutral", "age": "middle_aged"},
    ]

    for criteria in test_criteria:
        voice = interface.get_voice_by_criteria(**criteria)
        if voice:
            print(f"   ✅ {criteria} → {voice['name']} ({voice['voice_id'][:8]}...)")
        else:
            print(f"   ❌ {criteria} → Nenhuma voz encontrada")


if __name__ == "__main__":
    print("🎵 TESTE DAS VOZES AMERICANAS ELEVENLABS")
    print("=" * 50)

    # Executar testes
    asyncio.run(test_american_voices())
    asyncio.run(test_voice_selection())
