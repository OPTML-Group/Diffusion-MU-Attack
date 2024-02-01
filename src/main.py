from fastapi import FastAPI, status
from pydantic import BaseModel

app = FastAPI()


class UdiffInput(BaseModel):
    diffusion_model_id: str
    concept: str
    attacker: str

@app.post("/udiff/")
def run_attack(udiff_input : UdiffInput):
    print("="*200)
    from attack import Main

    results_dir = None
    config_file = None
    if udiff_input.concept == "nudity":
        if udiff_input.attacker == "text_grad":
            config_file = "/home/jiqingfe/Diffusion-MU-Attack/configs/nudity/text_grad_esd_nudity_classifier.json"
            results_dir = "/home/jiqingfe/Diffusion-MU-Attack/files/results/text_grad_esd_nudity_classifier/attack_idx_7/log.json"
        else:
            raise NotImplementedError("Not implemented yet")
    else:
        raise NotImplementedError("Not implemented yet")

    Main(config_file=config_file, model_name_or_path=udiff_input.model_id)

    f = open(results_dir)
    data = json.load(f)
    if data[-1]["success"]:
        result = data[-1]['prompt']
    else:
        result = "Attack failed"

    return {"prompt": result}