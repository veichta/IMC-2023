from hloc import extract_features, match_features

configs = {
    "sift+NN": {
        "features": {
            "model": {"name": "dog"},
            "options": {
                "first_octave": -1,
                "peak_threshold": 0.01,
            },
            "output": "feats-sift",
            "preprocessing": {"grayscale": True, "resize_max": 1600},
        },
        "matches": match_features.confs["NN-ratio"],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "loftr20": {
        "features": None,
        "matches": {
            "output": "matches-loftr",
            "model": {"name": "loftr", "weights": "outdoor"},
            "preprocessing": {"grayscale": True, "resize_max": 840, "dfactor": 8},  # 1024,
            "max_error": 1,  # max error for assigned keypoints (in px)
            "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 20,
    },
    "loftr50": {
        "features": None,
        "matches": {
            "output": "matches-loftr",
            "model": {"name": "loftr", "weights": "outdoor"},
            "preprocessing": {"grayscale": True, "resize_max": 840, "dfactor": 8},  # 1024,
            "max_error": 1,  # max error for assigned keypoints (in px)
            "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "SP+SG": {
        "features": extract_features.confs["superpoint_max"],
        "matches": match_features.confs["superglue"],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "SP+LG": {
        "features": extract_features.confs["superpoint_max"],
        "matches": {
            "output": "matches-sp-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "superpoint_lightglue",
                "flash": False,
                "filter_threshold": 0.1,
            },
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "SP+LG-flash": {
        "features": extract_features.confs["superpoint_max"],
        "matches": {
            "output": "matches-sp-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "superpoint_lightglue",
                "flash": True,
                "filter_threshold": 0.1,
            },
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "SP+LG+exhaustive": {
        "features": extract_features.confs["superpoint_max"],
        "matches": {
            "output": "matches-sp-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "superpoint_lightglue",
                "flash": False,
                "filter_threshold": 0.1,
            },
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 200,
    },
    "DISK+LG": {
        "features": extract_features.confs["disk"],
        "matches": {
            "output": "matches-disk-lightglue",
            "model": {
                "name": "lightglue",
                "weights": "disk_lightglue_legacy",
                "input_dim": 128,
                "flash": False,
                "filter_threshold": 0.1,
                "rotary": {
                    "axial": True,
                },
            },
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "SP+SG+cos": {
        "features": extract_features.confs["superpoint_max"],
        "matches": match_features.confs["superglue"],
        "retrieval": extract_features.confs["cosplace"],
        "n_retrieval": 50,
    },
    "disk": {
        "features": extract_features.confs["disk"],
        "matches": match_features.confs["NN-ratio"],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "DISK+LG+sift+NN": {
        "features": [extract_features.confs["disk"], extract_features.confs["sift"]],
        "matches": [
            {
                "output": "matches-disk-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "disk_lightglue_legacy",
                    "input_dim": 128,
                    "flash": False,
                    "filter_threshold": 0.1,
                    "rotary": {
                        "axial": True,
                    },
                },
            },
            match_features.confs["NN-ratio"],
        ],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "SP+LG+sift+NN": {
        "features": [extract_features.confs["superpoint_max"], extract_features.confs["sift"]],
        "matches": [
            {
                "output": "matches-sp-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "superpoint_lightglue",
                    "flash": False,
                    "filter_threshold": 0.1,
                },
            },
            match_features.confs["NN-ratio"],
        ],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "DISK+SP+LG": {
        "features": [extract_features.confs["disk"], extract_features.confs["superpoint_max"]],
        "matches": [
            {
                "output": "matches-disk-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "disk_lightglue_legacy",
                    "input_dim": 128,
                    "flash": False,
                    "filter_threshold": 0.1,
                    "rotary": {
                        "axial": True,
                    },
                },
            },
            {
                "output": "matches-sp-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "superpoint_lightglue",
                    "flash": False,
                    "filter_threshold": 0.1,
                },
            },
        ],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 30,
    },
    "DISK+SPv2+LG": {
        "features": [
            extract_features.confs["disk"],
            {
                "output": "feats-superpointv2-n4096-r1600",
                "model": {
                    "name": "superpoint_v2",
                    "max_num_keypoints": 4096,
                    "weights": "sp_caps",
                },
                "preprocessing": {
                    "resize_max": 1600,
                    "resize_force": True,
                },
            },
        ],
        "matches": [
            {
                "output": "matches-disk-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "disk_lightglue_legacy",
                    "input_dim": 128,
                    "flash": False,
                    "filter_threshold": 0.1,
                    "rotary": {
                        "axial": True,
                    },
                },
            },
            {
                "output": "matches-sp2-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "superpointv2_lightglue",
                    "input_dim": 128,
                    "flash": False,
                    "filter_threshold": 0.1,
                },
            },
        ],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 30,
    },
    "SPv2+LG+sift+NN": {
        "features": [
            {
                "output": "feats-superpointv2-n4096-r1600",
                "model": {
                    "name": "superpoint_v2",
                    "max_num_keypoints": 4096,
                    "weights": "sp_caps",
                },
                "preprocessing": {
                    "resize_max": 1600,
                    "resize_force": True,
                },
            },
            extract_features.confs["sift"],
        ],
        "matches": [
            {
                "output": "matches-sp2-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "superpointv2_lightglue",
                    "input_dim": 128,
                    "flash": False,
                    "filter_threshold": 0.1,
                },
            },
            match_features.confs["NN-ratio"],
        ],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "SPv2+LG": {
        "features": {
            "output": "feats-superpointv2-n4096-r1600",
            "model": {
                "name": "superpoint_v2",
                "max_num_keypoints": 4096,
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
                "flash": False,
                "filter_threshold": 0.1,
            },
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "alikedn16": {
        "features": {
            "output": "feats-alikedn16",
            "model": {
                "name": "aliked",
                'model_name': 'aliked-n16',  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                'max_num_keypoints': 4096,
                'detection_threshold': 0.0,
                'force_num_keypoints': False,
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
                "flash": False,
                "filter_threshold": 0.1,
            },
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "aliked+SPv2+LG": {
        "features": [
            {
                "output": "feats-superpointv2-n4096-r1600",
                "model": {
                    "name": "superpoint_v2",
                    "max_num_keypoints": 4096,
                    "weights": "sp_caps",
                },
                "preprocessing": {
                    "resize_max": 1600,
                    "resize_force": True,
                },
            },
            {
                "output": "feats-alikedn16",
                "model": {
                    "name": "aliked",
                    'model_name': 'aliked-n16',  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
                    'max_num_keypoints': 4096,
                    'detection_threshold': 0.0,
                    'force_num_keypoints': False,
                },
                "preprocessing": {
                    "resize_max": 1600,
                    # "resize_force": True,
                },
            },
        ],
        "matches": [
            {
                "output": "matches-sp2-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "superpointv2_lightglue",
                    "input_dim": 128,
                    "flash": False,
                    "filter_threshold": 0.1,
                },
            },
            {
                "output": "matches-aliked-lightglue",
                "model": {
                    "name": "lightglue",
                    "weights": "aliked_lightglue",
                    "input_dim": 128,
                    "flash": False,
                    "filter_threshold": 0.1,
                },
            },
        ],
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
}
