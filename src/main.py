from fastapi import FastAPI, status
from pydantic import BaseModel
import json

app = FastAPI()


class UdiffInput(BaseModel):
    diffusion_model_id: str
    concept: str
    attacker: str

@app.post("/udiff/")
def run_attack(udiff_input : UdiffInput):
    print("="*200)
    from execs.attack import Main

    results_dir = None
    config_file = None
    key = udiff_input.diffusion_model_id + "-" + udiff_input.concept + "-" + udiff_input.attacker
    with open("./results.json", "r") as f:
        json_data = json.load(f)
    if key in json_data.keys():
        print("result ====================================")
        print(json_data[key])
        return json_data[key]

    if udiff_input.concept == "nudity":
        if udiff_input.attacker == "text_grad":
            config_file = "/home/ubuntu/jiqing/Diffusion-MU-Attack/configs/nudity/text_grad_esd_nudity_classifier.json"
            results_dir = "/home/ubuntu/jiqing/Diffusion-MU-Attack/files/results/text_grad_esd_nudity_classifier/attack_idx_7/log.json"
        else:
            raise NotImplementedError("Nudity Attack Method Not implemented yet")
    elif udiff_input.concept == "object":
        if udiff_input.attacker == "text_grad":
            config_file = "/home/jiqingfe/Diffusion-MU-Attack/configs/object/text_grad_esd_church_classifier.json"
            results_dir = "/home/jiqingfe/Diffusion-MU-Attack/files/results/text_grad_esd_church_classifier/attack_idx_7/log.json"
        else:
            raise NotImplementedError("Object Attack Method Not implemented yet")
    elif udiff_input.concept == "style":
        if udiff_input.attacker == "text_grad":
            config_file = "/home/jiqingfe/Diffusion-MU-Attack/configs/style/text_grad_esd_vangogh_classsifier.json"
            results_dir = "/home/jiqingfe/Diffusion-MU-Attack/files/results/text_grad_esd_vangogh_classsifier/attack_idx_7/log.json"
        else:
            raise NotImplementedError("Style Attack Method Not implemented yet")
    else:
        raise NotImplementedError("Attack Concept Not implemented yet")

    print("Main start ==============================================================================")
    Main(config_file=config_file, model_name_or_path=udiff_input.diffusion_model_id)

    print("Main completed ==============================================================================")
    f = open(results_dir)
    data = json.load(f)
    if data[-1]["success"]:
        result = data[-1]['prompt']
    else:
        result = "Attack failed"

    print("result ====================================")
    print(result)

    json_data[key] = result
    with open("./results.json", "w") as f:
        json.dump(json_data, f)

    return result