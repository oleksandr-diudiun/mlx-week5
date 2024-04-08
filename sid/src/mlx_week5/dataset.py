import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, collated_data_dict, device="cpu"):
        self.collated_data = collated_data_dict
        self.device = device

    def __len__(self):
        return len(self.collated_data)

    def __getitem__(self, idx):
        return {
            "story_tokens": self.collated_data[idx][
                "ground_truth_labels_grid_tensor"
            ].to(self.device),

            "len_story_tokens": self.collated_data[idx][
                "len_story_tokens"
            ].to(self.device),
           
        }

    # Now pad the existing_objects_coordinates_tensor and return them as well as the length.
    @staticmethod
    def pad_helper(tensor):
        return torch.nn.utils.rnn.pad_sequence(
            tensor, batch_first=True, padding_value=-1
        )

    @staticmethod
    def collate_fn_wrapper(device="cpu"):
        def collate_fn(batch):

            image_ids = torch.tensor([item["image_id"] for item in batch]).to(device)
            images = torch.stack([item["image"] for item in batch]).to(device)
            ground_truth_labels_grid_tensor = torch.stack(
                [item["ground_truth_labels_grid_tensor"] for item in batch]
            ).to(device)
            existing_objects_coordinates_tensor = CustomDataset.pad_helper(
                    [item["existing_objects_coordinates_tensor"] for item in batch]
                ).to(device)
            non_existing_object_coordinates_tensor = CustomDataset.pad_helper(
                    [item["non_existing_object_coordinates_tensor"] for item in batch]
            ).to(device)
            len_existing_objects_coordinates_tensor = torch.stack(
                [item["len_existing_objects_coordinates_tensor"] for item in batch]
            ).to(device)
            len_non_existing_object_coordinates_tensor = torch.stack(
                [item["len_non_existing_object_coordinates_tensor"] for item in batch]
            ).to(device)

            return {
                "image_id": image_ids,
                "image": images,
                "ground_truth_labels_grid_tensor": ground_truth_labels_grid_tensor,
                "existing_objects_coordinates_tensor": existing_objects_coordinates_tensor,
                "non_existing_object_coordinates_tensor": non_existing_object_coordinates_tensor,
                "len_existing_objects_coordinates_tensor": len_existing_objects_coordinates_tensor,
                "len_non_existing_object_coordinates_tensor": len_non_existing_object_coordinates_tensor,
            }

        return collate_fn

    @staticmethod
    def unpad(padded_sequences, lengths):
        unpadded_sequences = []
        for i, length in enumerate(lengths):
            unpadded_sequences.append(padded_sequences[i, :length])
        return unpadded_sequences
