#!/usr/bin/env python3
"""
Remove duplicatas do arquivo de texto
"""


def clean_duplicates(input_file: str, output_file: str = None):
    """Remove duplicatas mantendo a primeira ocorrência."""
    if output_file is None:
        output_file = input_file

    # Ler todas as linhas
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remover duplicatas mantendo ordem
    seen = set()
    unique_lines = []
    duplicates_found = []

    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line and cleaned_line not in seen:
            seen.add(cleaned_line)
            unique_lines.append(line)
        elif cleaned_line in seen:
            duplicates_found.append(cleaned_line)

    # Escrever arquivo limpo
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(unique_lines)

    print(f"✅ Arquivo limpo salvo: {output_file}")
    print(f"📊 Estatísticas:")
    print(f"   • Original: {len(lines)} linhas")
    print(f"   • Limpo: {len(unique_lines)} linhas")
    print(f"   • Duplicatas removidas: {len(lines) - len(unique_lines)}")

    if duplicates_found:
        print(f"\n🔄 Duplicatas removidas:")
        for dup in duplicates_found:
            print(f"   • {dup[:80]}...")


if __name__ == "__main__":
    clean_duplicates("english-text.txt", "english-text-clean.txt")
