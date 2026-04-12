import torch
import numpy as np
from tqdm import tqdm


def extract_logits(model, dataloader, device):
    all_logits = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting logits", leave=False):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def compute_softmax_score(logits):
    logits_tensor = torch.from_numpy(logits).float()
    softmax_probs = torch.softmax(logits_tensor, dim=1)
    max_softmax, _ = torch.max(softmax_probs, dim=1)

    return max_softmax.numpy()


def compute_energy_score(logits, temperature=1.0):
    logits_tensor = torch.from_numpy(logits).float()
    neg_energy = temperature * torch.logsumexp(logits_tensor / temperature, dim=1)

    return neg_energy.numpy()


def compute_all_scores(model, dataloader, device, temperature=1.0):
    logits = extract_logits(model, dataloader, device)
    softmax_scores = compute_softmax_score(logits)
    energy_scores = compute_energy_score(logits, temperature=temperature)

    return {
        "logits": logits,
        "softmax_scores": softmax_scores,
        "energy_scores": energy_scores,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("Demo: OOD Detector Scoring")
    print("=" * 50)

    np.random.seed(42)
    fake_id_logits = np.random.randn(100, 10) + 2.0
    fake_ood_logits = np.random.randn(100, 10)

    id_softmax = compute_softmax_score(fake_id_logits)
    ood_softmax = compute_softmax_score(fake_ood_logits)
    print(f"\nSoftmax Score - ID  mean: {id_softmax.mean():.4f}")
    print(f"Softmax Score - OOD mean: {ood_softmax.mean():.4f}")

    id_energy = compute_energy_score(fake_id_logits, temperature=1.0)
    ood_energy = compute_energy_score(fake_ood_logits, temperature=1.0)
    print(f"\nEnergy Score  - ID  mean: {id_energy.mean():.4f}")
    print(f"Energy Score  - OOD mean: {ood_energy.mean():.4f}")