from hloc import extract_features, match_features

configs = {
    "SIFT": {
        "features": {
            "model": {"name": "dog"},
            "options": {
                "first_octave": -1,
                "peak_threshold": 0.00667,  # 0.00667, # 0.01,
            },
            "output": "feats-sift",
            "preprocessing": {"grayscale": True, "resize_max": 1600},
        },
        "matches": match_features.confs["NN-ratio"],
    },
    "loftr": {
        "features": None,
        "matches": {
            "output": "matches-loftr",
            "model": {"name": "loftr", "weights": "outdoor"},
            "preprocessing": {"grayscale": True, "resize_max": 840, "dfactor": 8},  # 1024,
            "max_error": 1,  # max error for assigned keypoints (in px)
            "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
        },
    },
    "SP+SG": {
        "features": extract_features.confs["superpoint_max"],
        "matches": match_features.confs["superglue"],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "DISK": {
        "features": {
            "output": "feats-disk",
            "model": {
                "name": "disk",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "matches": {
            "output": "matches-disk-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "disk_lightglue_legacy",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.1,
                "rotary": {
                    "axial": True,
                },
            },
        },
    },
    "DISKh": {
        "features": {
            "output": "feats-disk",
            "model": {
                "name": "disk",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "matches": {
            "output": "matches-disk-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "disk_lightglue_legacy",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.2,
                "rotary": {
                    "axial": True,
                },
            },
        },
    },
    "DISKl": {
        "features": {
            "output": "feats-disk",
            "model": {
                "name": "disk",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "matches": {
            "output": "matches-disk-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "disk_lightglue_legacy",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.01,
                "rotary": {
                    "axial": True,
                },
            },
        },
    },
    "DISK2K": {
        "features": {
            "output": "feats-disk2k",
            "model": {
                "name": "disk",
                "max_keypoints": 2048,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "matches": {
            "output": "matches-disk2k-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "disk_lightglue_legacy",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.1,
                "rotary": {
                    "axial": True,
                },
            },
        },
    },
    "SP": {
        "features": {
            "output": "feats-superpoint-n4096-rmax1600",
            "model": {
                "name": "superpoint",
                "nms_radius": 3,
                "max_keypoints": 4096,
            },
            "preprocessing": {
                "grayscale": True,
                "resize_max": 1600,
                "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-sp-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "superpoint_lightglue",
                "flash": True,
                "filter_threshold": 0.1,
            },
        },
    },
    "SPv2": {
        "features": {
            "output": "feats-superpointv2-n4096-r1600",
            "model": {
                "name": "superpoint_v2",
                "max_num_keypoints": 4096,
                "nms_radius": 8,
                "detection_threshold": 0.000,
                "weights": "sp_caps",
            },
            "preprocessing": {
                "resize_max": 1600,
                "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-sp2-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "superpointv2_lightglue",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.1,
            },
        },
    },
    "SPv2l": {
        "features": {
            "output": "feats-superpointv2-n4096-r1600",
            "model": {
                "name": "superpoint_v2",
                "max_num_keypoints": 4096,
                "nms_radius": 8,
                "detection_threshold": 0.000,
                "weights": "sp_caps",
            },
            "preprocessing": {
                "resize_max": 1600,
                "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-sp2-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "superpointv2_lightglue",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.01,
            },
        },
    },
    "SPv2h": {
        "features": {
            "output": "feats-superpointv2-n4096-r1600",
            "model": {
                "name": "superpoint_v2",
                "max_num_keypoints": 4096,
                "nms_radius": 8,
                "detection_threshold": 0.000,
                "weights": "sp_caps",
            },
            "preprocessing": {
                "resize_max": 1600,
                "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-sp2-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "superpointv2_lightglue",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.2,
            },
        },
    },
    "SPv2S": {
        "features": {
            "output": "feats-superpointv2-n4096-r1024",
            "model": {
                "name": "superpoint_v2",
                "max_num_keypoints": 4096,
                "nms_radius": 8,
                "detection_threshold": 0.000,
                "weights": "sp_caps",
            },
            "preprocessing": {
                "resize_max": 1024,
                "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-sp2-lightglue-S",
            "model": {
                "name": "lightglue",
                "weights": "superpointv2_lightglue",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.1,
            },
        },
    },
    "SPv2L": {
        "features": {
            "output": "feats-superpointv2-n4096-r1920",
            "model": {
                "name": "superpoint_v2",
                "max_num_keypoints": 4096,
                "nms_radius": 8,
                "detection_threshold": 0.000,
                "weights": "sp_caps",
            },
            "preprocessing": {
                "resize_max": 1920,
                "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-sp2-lightglue-L",
            "model": {
                "name": "lightglue",
                "weights": "superpointv2_lightglue",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.1,
            },
        },
    },
    "ALIKED": {
        "features": {
            "output": "feats-alikedn16",
            "model": {
                "name": "aliked",
                "model_name": "aliked-n16",  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                "max_num_keypoints": 4096,
                "detection_threshold": 0.0,
                "force_num_keypoints": False,
            },
            "preprocessing": {
                "resize_max": 1600,
                # "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-aliked-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "aliked_lightglue",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.1,
            },
        },
    },
    "ALIKED2K": {
        "features": {
            "output": "feats-aliked2k",
            "model": {
                "name": "aliked",
                "model_name": "aliked-n16",  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                "max_num_keypoints": 2048,
                "detection_threshold": 0.0,
                "force_num_keypoints": False,
            },
            "preprocessing": {
                "resize_max": 1600,
                # "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-aliked2k-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "aliked_lightglue",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.1,
            },
        },
    },
    "ALIKED2Kh": {
        "features": {
            "output": "feats-aliked2k",
            "model": {
                "name": "aliked",
                "model_name": "aliked-n16",  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                "max_num_keypoints": 2048,
                "detection_threshold": 0.0,
                "force_num_keypoints": False,
            },
            "preprocessing": {
                "resize_max": 1600,
                # "resize_force": True,
            },
        },
        "matches": {
            "output": "matches-aliked2k-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "aliked_lightglue",
                "input_dim": 128,
                "flash": True,
                "filter_threshold": 0.2,
            },
        },
        "ALIKED2Kl": {
            "features": {
                "output": "feats-aliked2k",
                "model": {
                    "name": "aliked",
                    "model_name": "aliked-n16",  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                    "max_num_keypoints": 2048,
                    "detection_threshold": 0.0,
                    "force_num_keypoints": False,
                },
                "preprocessing": {
                    "resize_max": 1600,
                    # "resize_force": True,
                },
            },
            "matches": {
                "output": "matches-aliked2k-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "aliked_lightglue",
                    "input_dim": 128,
                    "flash": True,
                    "filter_threshold": 0.01,
                },
            },
        },
    },
}
