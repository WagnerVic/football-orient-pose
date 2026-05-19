"""Interfaces para estimadores de pose corporal."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from football_orient_pose.utils.keypoint_mapping import coco17_to_h3wb17


class BasePoseEstimator(ABC):
    """Interface comum para estimadores de pose 2D.

    Estimadores concretos como RTMPose, HRNet e OpenPose devem implementar
    inferência em formato COCO-17. A conversão para H3WB-17 fica centralizada
    nesta classe base para manter o pipeline independente do modelo escolhido.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificável do estimador."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Estima keypoints de uma imagem.

        Parameters
        ----------
        image : np.ndarray
            Imagem de entrada, normalmente em formato ``H x W x C``.

        Returns
        -------
        np.ndarray
            Array de shape ``(17, 3)`` no formato COCO-17, com colunas
            ``x``, ``y`` e ``confidence``.
        """

    @abstractmethod
    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Estima keypoints de um lote de imagens.

        Parameters
        ----------
        images : list[np.ndarray]
            Lista de imagens de entrada.

        Returns
        -------
        np.ndarray
            Array de shape ``(N, 17, 3)`` no formato COCO-17, onde ``N`` é
            o número de imagens.
        """

    def predict_h3wb(self, image: np.ndarray) -> np.ndarray:
        """Estima keypoints e converte o resultado para H3WB-17.

        Parameters
        ----------
        image : np.ndarray
            Imagem de entrada, normalmente em formato ``H x W x C``.

        Returns
        -------
        np.ndarray
            Array de shape ``(17, 2)`` no formato H3WB-17, contendo apenas
            coordenadas ``x`` e ``y``.
        """
        coco_keypoints = self.predict(image)
        return coco17_to_h3wb17(coco_keypoints[:, :2])
