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
}
