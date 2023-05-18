from hloc import extract_features, match_features

configs = {
    "sift+NN": {
        "features": extract_features.confs["sift"],
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
    "dkm-full": {
        "features": None,
        "matches": {
            "output": "matches-dkm",
            "model": {"name": "dkm", "weights": "outdoor"},
            "preprocessing": {"grayscale": False, "resize_max": 1024, "dfactor": 8},  # 1024,
            "max_error": 2,  # max error for assigned keypoints (in px)
            "cell_size": 2,  # size of quantization patch (max 1 kp/patch)
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 50,
    },
    "dkm-lim": {
        "features": None,
        "matches": {
            "output": "matches-dkm",
            "model": {
                "name": "dkm",
                "weights": "indoor",
                "sample_num": 5000,
                "max_num_matches": None,
            },
            "preprocessing": {"grayscale": False, "resize_max": 1024, "dfactor": 8},  # 1024,
            "max_error": 4,  # max error for assigned keypoints (in px)
            "cell_size": 2,  # size of quantization patch (max 1 kp/patch)
        },
        "retrieval": extract_features.confs["netvlad"],
        "n_retrieval": 20,
    },
}
