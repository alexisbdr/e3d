from dataclasses import dataclass

import esim_py
import numpy as np


@dataclass
class EsimParams:

    Cp: float = 0.5
    Cn: float = 0.5
    sigma_cp: float = 0.03
    sigma_cn: float = 0.03
    refractory_period: float = 1e-4
    log_eps: float = 1e-3
    use_log: bool = True

    show_frame: bool = False


def generate_event_frames(image_path_list, img_size, batch):

    event_frames = []
    curr_batch = 0
    while curr_batch < len(image_path_list) / batch:

        batch_image_path_list = image_path_list[
            curr_batch * batch : (curr_batch + 1) * batch
        ]

        timestamp_list = range(len(batch_image_path_list))
        # print("timestamp list: ", timestamp_list)
        esim = esim_py.EventSimulator(
            EsimParams.Cp,
            EsimParams.Cn,
            EsimParams.refractory_period,
            EsimParams.log_eps,
            EsimParams.use_log,
            EsimParams.sigma_cp,
            EsimParams.sigma_cn,
        )

        events = esim.generateFromStampedImageSequence(
            batch_image_path_list, timestamp_list
        )

        batch_events_image = int(len(events) / len(batch_image_path_list))
        # print("batch events image size: ",batch_events_image)
        event_batch_size = 0
        while event_batch_size <= len(events) and len(event_frames) < len(
            image_path_list
        ):

            curr_batch_events = events[
                event_batch_size : event_batch_size + batch_events_image
            ]

            pos_events = curr_batch_events[curr_batch_events[:, -1] == 1]
            neg_events = curr_batch_events[curr_batch_events[:, -1] == -1]

            image_pos = np.zeros(img_size[0] * img_size[1], dtype="uint8")
            image_neg = np.zeros(img_size[0] * img_size[1], dtype="uint8")

            np.add.at(
                image_pos,
                (pos_events[:, 0] + pos_events[:, 1] * img_size[1]).astype("int32"),
                pos_events[:, -1] ** 2,
            )
            np.add.at(
                image_neg,
                (neg_events[:, 0] + neg_events[:, 1] * img_size[1]).astype("int32"),
                neg_events[:, -1] ** 2,
            )

            image_rgb = (
                np.stack(
                    [
                        image_pos.reshape(img_size),
                        np.zeros(img_size, dtype="uint8"),
                        image_neg.reshape(img_size),
                    ],
                    -1,
                )
                * 50
            )

            # img_black = np.all(image_rgb == [0,0,0], axis=-1)
            # image_rgb[img_black] = [255, 255, 255]

            event_frames.append(image_rgb)

            if EsimParams.show_frame:
                plt.imshow(image_rgb)
                plt.show()

            # event_batch_size += batch_events_plot
            event_batch_size += len(curr_batch_events)

        curr_batch += 1

    return event_frames
