# Navegación Autónoma de Embarcaciones de Superficie con Reinforcement Learning

Proyecto de investigación aplicada en aprendizaje por refuerzo para la navegación autónoma de embarcaciones de superficie en entornos marítimos dinámicos, usando CommonOcean-Sim como simulador base.

El objetivo principal es entrenar una política de control capaz de gobernar una embarcación ego, evitar colisiones con tráfico marítimo dinámico y mejorar el comportamiento de navegación mediante randomización de dominio y preferencias inspiradas en COLREGs.

## Objetivo

Desarrollar un agente de Reinforcement Learning que controle una embarcación de superficie en escenarios marítimos con obstáculos dinámicos.

El agente controla únicamente la embarcación ego. Las demás embarcaciones se modelan como parte del entorno mediante trayectorias predefinidas de tipo `dynamicObstacle`.

El proyecto estudia tres niveles progresivos de desempeño:

1. navegación nominal sin colisión;
2. generalización frente a variaciones del escenario mediante Domain Randomization;
3. mejora de márgenes de seguridad mediante una recompensa auxiliar inspirada en COLREGs.

## Simulador

El proyecto utiliza:

- CommonOcean-Sim;
- escenarios CommonOcean en formato XML;
- modelo dinámico de embarcación de superficie tipo YP;
- obstáculos dinámicos para representar tráfico marítimo;
- detección de colisiones del simulador;
- entorno Gymnasium personalizado para entrenamiento RL.

Repositorio base del simulador:

https://github.com/CommonOcean/commonocean-sim

## Arquitectura de control

El controlador MPC original de CommonOcean-Sim para la embarcación ego fue reemplazado por un controlador externo conectado a una política RL:

```text
PPO Policy
    -> CommonOceanEnv
    -> ExternalActionController
    -> SurfaceVessel YP dynamics
    -> CommonOcean-Sim
