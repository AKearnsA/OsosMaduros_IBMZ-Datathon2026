from explain.schemas import ExplanationRequest, ExplanationResponse


class RiskExplainer:
    def __init__(self):
        # Mapeo de recomendaciones según factores de riesgo
        self.action_matrix = {
            "nonsync_share": "Monitorear la inercia del sistema y preparar reservas síncronas rápidas.",
            "interconnection_balance_mw": "Evaluar la capacidad del enlace de interconexión y ajustar flujos transfronterizos.",
            "inertia_est_gws": "Activar capacidad de regulación primaria o inercia sintética si está disponible."
        }

    def explain(self, data: ExplanationRequest) -> ExplanationResponse:
        # 1. Resumen según el nivel de riesgo
        if data.level == "Critical":
            summary = f"ALERTA CRÍTICA: Puntuación de riesgo de {data.risk_score:.2f}. Se requiere intervención operativa inmediata en el sistema."
        elif data.level == "Watch":
            summary = f"AVISO DE PRECAUCIÓN: Puntuación de riesgo elevada ({data.risk_score:.2f}). Se detecta una tendencia hacia la inestabilidad de frecuencia."
        else:
            summary = f"ESTADO ESTABLE: El sistema opera dentro de parámetros de seguridad (Riesgo: {data.risk_score:.2f})."

        # 2. Recomendaciones basadas en los factores identificados
        actions = []
        for factor in data.top_factors:
            if factor in self.action_matrix:
                actions.append(self.action_matrix[factor])
        
        if not actions:
            actions.append("Mantener la monitorización rutinaria de la telemetría del sistema.")

        # 3. Detalle técnico
        factors_str = ", ".join(data.top_factors)
        technical_details = (
            f"Evento registrado a las {data.timestamp}. "
            f"Factores determinantes: [{factors_str}]. "
            f"Simulación: {'Sí' if data.is_simulated else 'No'}."
        )

        return ExplanationResponse(
            summary=summary,
            recommended_actions=actions,
            technical_details=technical_details
        )