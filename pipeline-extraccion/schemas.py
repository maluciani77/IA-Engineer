"""Contrato de datos del pipeline (Pydantic).

Las `description` de cada campo no son comentarios: LangChain se las envía
al modelo junto con el esquema, así que funcionan como instrucciones.
"""

from enum import Enum

from pydantic import BaseModel, Field


class NivelCriticidad(str, Enum):
    """Lista cerrada de valores permitidos para la criticidad."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Información técnica extraída de un texto libre."""

    tecnologias: list[str] = Field(
        min_length=1,  # restricción: la lista no puede venir vacía
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos o servicios "
            "mencionados explícitamente en el texto"
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "Gravedad o urgencia de la situación descrita: 'baja' (informativo), "
            "'media' (degradación o riesgo) o 'alta' (caída, pérdida de datos, "
            "impacto en usuarios)"
        ),
    )
    resumen_tecnico: str = Field(
        min_length=10,  # restricción: evita resúmenes vacíos o de una palabra
        description="Resumen técnico breve del texto, en una o dos oraciones",
    )
