
from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "drone_full_demo.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("\n===== MUJOCO COLLISION GEOMETRY CHECK =====")

print(f"Number of model geoms: {model.ngeom}")
print(f"Number of simulation bodies: {model.nbody}")

print("\nGeometries in the XML model:")

for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
    )
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, body_id
    )

    print(
        f"Geom ID: {geom_id}, "
        f"name: {name or '(unnamed)'}, "
        f"body: {body_name or '(unnamed)'}, "
        f"collision type: {model.geom_contype[geom_id]}, "
        f"collision affinity: {model.geom_conaffinity[geom_id]}"
    )

print("\nChecking contacts at the model's initial position...")

mujoco.mj_forward(model, data)

print(f"Initial contact count: {data.ncon}")

for contact_id in range(data.ncon):
    contact = data.contact[contact_id]

    geom1 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
    )
    geom2 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
    )

    print(
        f"Contact {contact_id}: "
        f"{geom1 or contact.geom1} <-> "
        f"{geom2 or contact.geom2}, "
        f"distance={contact.dist:.5f} m"
    )

print("\nThis checks the initial pose only.")
print("It does not yet test collisions throughout the flight.")
