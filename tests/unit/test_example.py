"""
Teste de exemplo para verificar se a estrutura está funcionando.

Este arquivo será removido quando implementarmos os testes reais.
"""

import pytest


def test_basic_math():
    """Teste básico para verificar se pytest está funcionando."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6


def test_string_operations():
    """Teste básico de operações com strings."""
    text = "ValeTTS"
    assert text.lower() == "valetts"
    assert len(text) == 7


@pytest.mark.parametrize("input_val,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
])
def test_parametrized_double(input_val, expected):
    """Teste parametrizado de exemplo."""
    assert input_val * 2 == expected


class TestExampleClass:
    """Classe de teste de exemplo."""
    
    def test_list_operations(self):
        """Teste de operações com listas."""
        test_list = [1, 2, 3]
        test_list.append(4)
        assert len(test_list) == 4
        assert test_list[-1] == 4
    
    def test_dict_operations(self):
        """Teste de operações com dicionários."""
        test_dict = {"key": "value"}
        test_dict["new_key"] = "new_value"
        assert "key" in test_dict
        assert test_dict["new_key"] == "new_value"


@pytest.mark.slow
def test_slow_operation():
    """Teste marcado como lento."""
    import time
    time.sleep(0.1)  # Simula operação lenta
    assert True


def test_exception_handling():
    """Teste de tratamento de exceções."""
    with pytest.raises(ZeroDivisionError):
        result = 1 / 0 