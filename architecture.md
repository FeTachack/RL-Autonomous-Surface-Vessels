# Arquitectura inicial

## Propiedad del control

Sea `ego = simulator.models[ego_index]`.

Al construir un episodio:

```python
ego.set_controller(ExternalActionController(...))
```

No se modifica ningún otro elemento de `simulator.models`.

Los obstáculos dinámicos del escenario no son convertidos en agentes ni reciben
acciones desde Gymnasium.

## Un paso

```text
1. RL emite a_norm en [-1, 1]^2.
2. YawActionScaler produce [a_n, yaw_rate].
3. ExternalActionController almacena la acción.
4. Simulator.compute_next_state() consulta cada controlador:
   - ego recibe la acción RL;
   - los demás modelos usan sus controladores originales.
5. CommonOcean actualiza estados y listeners.
6. El wrapper calcula observación, recompensa y terminación.
```

## Límites deliberados

- La detección de colisión todavía está detrás de `CollisionProbe`.
- La observación de tráfico tendrá su propio builder.
- Los valores físicos de acción se declaran en YAML hasta confirmar y validar
  los parámetros exactos del tipo de embarcación seleccionado.
- `reset()` reconstruye la simulación completa.
