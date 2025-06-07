# Sistema de Aprovação Humana para Mudanças LLM
# Permitir interação durante o treinamento

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Regra de validação para parâmetros."""

    param_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    value_type: type = float
    valid_values: Optional[List[Any]] = None


class HumanApprovalSystem:
    """Sistema de aprovação humana para mudanças críticas do LLM."""

    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        self.validation_rules = self._setup_validation_rules()

    def _setup_validation_rules(self) -> Dict[str, ValidationRule]:
        """Define regras de validação para cada parâmetro."""
        return {
            "learning_rate": ValidationRule(
                param_name="learning_rate",
                min_value=1e-6,
                max_value=1e-1,
                value_type=float,
            ),
            "batch_size": ValidationRule(
                param_name="batch_size", min_value=1, max_value=512, value_type=int
            ),
            "mel_loss_weight": ValidationRule(
                param_name="mel_loss_weight",
                min_value=0.1,
                max_value=100.0,
                value_type=float,
            ),
            "kl_loss_weight": ValidationRule(
                param_name="kl_loss_weight",
                min_value=0.0,
                max_value=10.0,
                value_type=float,
            ),
            "adv_loss_weight": ValidationRule(
                param_name="adv_loss_weight",
                min_value=0.0,
                max_value=10.0,
                value_type=float,
            ),
            "fm_loss_weight": ValidationRule(
                param_name="fm_loss_weight",
                min_value=0.0,
                max_value=10.0,
                value_type=float,
            ),
            "duration_loss_weight": ValidationRule(
                param_name="duration_loss_weight",
                min_value=0.0,
                max_value=10.0,
                value_type=float,
            ),
            "max_grad_norm": ValidationRule(
                param_name="max_grad_norm",
                min_value=0.1,
                max_value=100.0,
                value_type=float,
            ),
        }

    def request_approval(
        self,
        epoch: int,
        current_config: Dict[str, Any],
        suggested_changes: Dict[str, Any],
        llm_analysis: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Solicita aprovação humana para mudanças críticas.

        Args:
            epoch: Época atual
            current_config: Configuração atual
            suggested_changes: Mudanças sugeridas pelo LLM
            llm_analysis: Análise do LLM em texto

        Returns:
            (approved, final_config): Se aprovado e configuração final
        """
        print("\n" + "=" * 80)
        print("🤖 APROVAÇÃO HUMANA NECESSÁRIA - MUDANÇAS CRÍTICAS DETECTADAS")
        print("=" * 80)
        print(f"📍 Época: {epoch}")
        print(f"⏰ Tempo para responder: {self.timeout_seconds} segundos")

        if llm_analysis:
            print(f"\n📋 Análise do LLM:")
            print(f"   {llm_analysis}")

        print(f"\n🔄 Mudanças Sugeridas:")
        for param, new_value in suggested_changes.items():
            current_value = current_config.get(param, "N/A")
            print(f"   • {param}: {current_value} → {new_value}")

        print(f"\n⚠️  IMPORTANTE: Mudanças incorretas podem:")
        print(f"   • Parar o treinamento")
        print(f"   • Corromper o aprendizado")
        print(f"   • Desperdiçar tempo de GPU")

        # Input com timeout
        user_choice = self._get_user_input_with_timeout()

        if user_choice == "timeout":
            print(f"\n⏰ Timeout! Rejeitando mudanças por segurança.")
            return False, current_config

        if user_choice.lower() in ["n", "no", "nao", "não"]:
            print(f"\n❌ Mudanças rejeitadas pelo usuário.")
            return False, current_config

        if user_choice.lower() in ["y", "yes", "sim", "s"]:
            # Aplicar mudanças sugeridas
            final_config = current_config.copy()
            final_config.update(suggested_changes)
            print(f"\n✅ Mudanças aprovadas!")
            return True, final_config

        if user_choice.lower() in ["c", "custom", "personalizar"]:
            # Permitir customização
            final_config = self._customize_parameters(current_config, suggested_changes)
            if final_config:
                print(f"\n✅ Configuração personalizada aprovada!")
                return True, final_config
            else:
                print(f"\n❌ Configuração personalizada cancelada.")
                return False, current_config

        print(f"\n❌ Opção inválida. Rejeitando mudanças por segurança.")
        return False, current_config

    def _get_user_input_with_timeout(self) -> str:
        """Obtém input do usuário com timeout."""
        print(f"\n🔧 Opções:")
        print(f"   [Y] Sim - Aplicar mudanças sugeridas")
        print(f"   [N] Não - Rejeitar mudanças")
        print(f"   [C] Personalizar - Editar manualmente")

        user_input = [None]

        def get_input():
            try:
                user_input[0] = input(f"\nEscolha [Y/N/C]: ").strip()
            except (EOFError, KeyboardInterrupt):
                user_input[0] = "n"

        # Thread para input com timeout
        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True
        input_thread.start()

        # Countdown
        for remaining in range(self.timeout_seconds, 0, -1):
            if user_input[0] is not None:
                break
            print(f"\r⏰ Tempo restante: {remaining:2d}s", end="", flush=True)
            time.sleep(1)

        input_thread.join(timeout=0.1)
        print()  # Nova linha

        return user_input[0] if user_input[0] is not None else "timeout"

    def _customize_parameters(
        self, current_config: Dict[str, Any], suggested_changes: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Permite customização manual dos parâmetros."""
        print(f"\n🛠️  MODO PERSONALIZAÇÃO")
        print(f"   Digite novos valores ou pressione Enter para manter")
        print(f"   Digite 'cancel' para cancelar")

        final_config = current_config.copy()

        for param, suggested_value in suggested_changes.items():
            current_value = current_config.get(param, "N/A")

            while True:
                print(f"\n📝 {param}:")
                print(f"   Atual: {current_value}")
                print(f"   Sugerido: {suggested_value}")

                # Mostrar regras de validação
                if param in self.validation_rules:
                    rule = self.validation_rules[param]
                    print(f"   Limites: {rule.min_value} - {rule.max_value}")
                    print(f"   Tipo: {rule.value_type.__name__}")

                user_input = input(
                    f"   Novo valor (Enter=manter, cancel=cancelar): "
                ).strip()

                if user_input.lower() == "cancel":
                    return None

                if not user_input:  # Enter = manter atual
                    break

                # Validar input
                is_valid, validated_value, error_msg = self._validate_parameter(
                    param, user_input
                )

                if is_valid:
                    final_config[param] = validated_value
                    print(f"   ✅ Valor aceito: {validated_value}")
                    break
                else:
                    print(f"   ❌ Erro: {error_msg}")
                    print(f"   Tente novamente...")

        # Confirmação final
        print(f"\n🔍 RESUMO DAS MUDANÇAS:")
        changes_made = False
        for param, new_value in final_config.items():
            old_value = current_config.get(param, "N/A")
            if old_value != new_value:
                print(f"   • {param}: {old_value} → {new_value}")
                changes_made = True

        if not changes_made:
            print(f"   Nenhuma mudança realizada.")
            return current_config

        confirmation = input(f"\n✅ Confirmar mudanças? [y/N]: ").strip().lower()

        if confirmation in ["y", "yes", "sim", "s"]:
            return final_config
        else:
            return None

    def _validate_parameter(
        self, param_name: str, value_str: str
    ) -> Tuple[bool, Any, str]:
        """
        Valida um parâmetro.

        Returns:
            (is_valid, validated_value, error_message)
        """
        if param_name not in self.validation_rules:
            return False, None, f"Parâmetro '{param_name}' não reconhecido"

        rule = self.validation_rules[param_name]

        try:
            # Converter para tipo correto
            if rule.value_type == int:
                value = int(value_str)
            elif rule.value_type == float:
                value = float(value_str)
            else:
                value = rule.value_type(value_str)

            # Verificar limites
            if rule.min_value is not None and value < rule.min_value:
                return False, None, f"Valor muito baixo. Mínimo: {rule.min_value}"

            if rule.max_value is not None and value > rule.max_value:
                return False, None, f"Valor muito alto. Máximo: {rule.max_value}"

            # Verificar valores válidos
            if rule.valid_values is not None and value not in rule.valid_values:
                return (
                    False,
                    None,
                    f"Valor inválido. Valores aceitos: {rule.valid_values}",
                )

            return True, value, ""

        except ValueError:
            return (
                False,
                None,
                f"Formato inválido. Esperado: {rule.value_type.__name__}",
            )
        except Exception as e:
            return False, None, f"Erro de validação: {str(e)}"
