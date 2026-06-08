"""Transforms customizados para o pipeline de augmentation do 3DSP.

Registrados no registry mmpose::transform via decorator, então basta adicionar
`football_orient_pose.finetuning.transforms` em `custom_imports` do config.

Motivação: `RandomErasing` e `Albu` não estão disponíveis no mmpose do container
(versão/dependências incompatíveis). Implementamos equivalentes diretos com numpy
e cv2, que já estão presentes no ambiente.
"""

import numpy as np
import cv2
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SimpleRandomErasing:
    """Simula oclusão aleatória apagando um patch da imagem cropada.

    Parâmetros idênticos ao RandomErasing do mmpose para compatibilidade:
        n_patches: (min, max) número de patches por imagem.
        ratio: (min_frac, max_frac) fração da área total a apagar.
        prob: probabilidade de aplicar o erasing.

    Preenche o patch com ruído uniforme (simula fundo ou equipamento ocluindo
    o membro — mais realista que preencher com zero/média).
    """

    def __init__(self, n_patches=(1, 1), ratio=(0.02, 0.10), prob=0.3):
        self.n_patches = n_patches
        self.ratio = ratio
        self.prob = prob

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results

        img = results["img"]
        h, w = img.shape[:2]
        n = np.random.randint(self.n_patches[0], self.n_patches[1] + 1)

        for _ in range(n):
            area_frac = np.random.uniform(self.ratio[0], self.ratio[1])
            area = h * w * area_frac
            aspect = np.random.uniform(0.5, 2.0)
            eh = int(np.clip(np.sqrt(area * aspect), 1, h))
            ew = int(np.clip(np.sqrt(area / aspect), 1, w))
            x = np.random.randint(0, max(1, w - ew + 1))
            y = np.random.randint(0, max(1, h - eh + 1))
            img[y:y + eh, x:x + ew] = np.random.randint(
                0, 256, (eh, ew, img.shape[2]), dtype=np.uint8
            )

        results["img"] = img
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(n_patches={self.n_patches}, "
                f"ratio={self.ratio}, prob={self.prob})")


@TRANSFORMS.register_module()
class SimpleMotionBlur:
    """Aplica motion blur direcional (horizontal ou vertical) com cv2.

    Simula o borrão de câmera em transmissões de futebol durante movimentos
    rápidos (panning, chute). Kernel 1D aplicado em direção aleatória.

    Parâmetros compatíveis com albumentations MotionBlur:
        blur_limit: (min_k, max_k) intervalo de tamanhos de kernel (ímpares).
        p: probabilidade de aplicar o blur.
    """

    def __init__(self, blur_limit=(3, 9), p=0.5):
        self.blur_limit = blur_limit
        self.p = p

    def __call__(self, results):
        if np.random.random() > self.p:
            return results

        img = results["img"]
        # Escolhe kernel ímpar no intervalo
        lo, hi = self.blur_limit
        ks = list(range(lo if lo % 2 == 1 else lo + 1, hi + 1, 2))
        if not ks:
            ks = [3]
        k = np.random.choice(ks)

        kernel = np.zeros((k, k), dtype=np.float32)
        if np.random.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k   # horizontal
        else:
            kernel[:, k // 2] = 1.0 / k   # vertical

        results["img"] = cv2.filter2D(img, -1, kernel)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(blur_limit={self.blur_limit}, "
                f"p={self.p})")
