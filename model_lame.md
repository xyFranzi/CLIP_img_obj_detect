## Current Issues Analysis

1. **Very small differences between good/failure similarities** (0.01-0.02) suggest the prompts might not be discriminative enough
2. **All objects scoring as "medium" quality** indicates the thresholds might need adjustment
3. **The difference between object types is larger than good vs failure within the same type** - this suggests the model is more sensitive to object category than quality
4. **"knive" (typo) returns 0.0 similarities** because it's not in the prompt dictionary

## POINT 3

### my thought

eg for one image the output is:
{'image': 'data/img/img__00016_.png', 'true_label': 'plate', 'quality_score': 0.4949, 'quality_category': 'medium', 'good_similarity': 0.2592, 'failure_similarity': 0.2795, 'confidence': 0.0202}
{'image': 'data/img/img__00016_.png', 'true_label': 'glass', 'quality_score': 0.5039, 'quality_category': 'medium', 'good_similarity': 0.2395, 'failure_similarity': 0.2238, 'confidence': 0.0157}
{'image': 'data/img/img__00016_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00016_.png', 'true_label': 'napkin', 'quality_score': 0.502, 'quality_category': 'medium', 'good_similarity': 0.2371, 'failure_similarity': 0.2292, 'confidence': 0.008}
{'image': 'data/img/img__00016_.png', 'true_label': 'bowl', 'quality_score': 0.5011, 'quality_category': 'medium', 'good_similarity': 0.2338, 'failure_similarity': 0.2292, 'confidence': 0.0046}
{'image': 'data/img/img__00016_.png', 'true_label': 'fork', 'quality_score': 0.5026, 'quality_category': 'medium', 'good_similarity': 0.3379, 'failure_similarity': 0.3275, 'confidence': 0.0104}
{'image': 'data/img/img__00016_.png', 'true_label': 'plate_dirt', 'quality_score': 0.5038, 'quality_category': 'medium', 'good_similarity': 0.298, 'failure_similarity': 0.2828, 'confidence': 0.0152}
{'image': 'data/img/img__00016_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.5007, 'quality_category': 'medium', 'good_similarity': 0.2578, 'failure_similarity': 0.2549, 'confidence': 0.0029}
i noticed the diffdrence of similarities between different items but not between good and failure of one item, does this difference make sense or can actually show something( like the image is realistic or not), or does this just depends on the prompt and makes no sense in reality
how can i further improve the code??

## Proof in the data

´´´
7%|█████▋                                                                           | 7/100 [00:38<08:01,  5.17s/it]{'image': 'data/img/img__00008_.png', 'true_label': 'fork', 'quality_score': 0.5003, 'quality_category': 'medium', 'good_similarity': 0.271, 'failure_similarity': 0.2699, 'confidence': 0.001}
{'image': 'data/img/img__00008_.png', 'true_label': 'glass', 'quality_score': 0.503, 'quality_category': 'medium', 'good_similarity': 0.2372, 'failure_similarity': 0.2254, 'confidence': 0.0118}
{'image': 'data/img/img__00008_.png', 'true_label': 'plate', 'quality_score': 0.4981, 'quality_category': 'medium', 'good_similarity': 0.2573, 'failure_similarity': 0.2649, 'confidence': 0.0076}
{'image': 'data/img/img__00008_.png', 'true_label': 'bowl', 'quality_score': 0.5004, 'quality_category': 'medium', 'good_similarity': 0.2459, 'failure_similarity': 0.2443, 'confidence': 0.0015}
{'image': 'data/img/img__00008_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00008_.png', 'true_label': 'napkin', 'quality_score': 0.5043, 'quality_category': 'medium', 'good_similarity': 0.2754, 'failure_similarity': 0.258, 'confidence': 0.0174}
{'image': 'data/img/img__00008_.png', 'true_label': 'plate_dirt', 'quality_score': 0.502, 'quality_category': 'medium', 'good_similarity': 0.2838, 'failure_similarity': 0.2758, 'confidence': 0.008}
{'image': 'data/img/img__00008_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.5033, 'quality_category': 'medium', 'good_similarity': 0.2639, 'failure_similarity': 0.2508, 'confidence': 0.0132}
  8%|██████▍                                                                          | 8/100 [00:41<07:08,  4.66s/it]{'image': 'data/img/img__00009_.png', 'true_label': 'glass', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.2591, 'failure_similarity': 0.2592, 'confidence': 0.0}
{'image': 'data/img/img__00009_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00009_.png', 'true_label': 'fork', 'quality_score': 0.5004, 'quality_category': 'medium', 'good_similarity': 0.3418, 'failure_similarity': 0.3404, 'confidence': 0.0014}
{'image': 'data/img/img__00009_.png', 'true_label': 'napkin', 'quality_score': 0.5081, 'quality_category': 'medium', 'good_similarity': 0.3516, 'failure_similarity': 0.319, 'confidence': 0.0325}
{'image': 'data/img/img__00009_.png', 'true_label': 'bowl', 'quality_score': 0.4998, 'quality_category': 'medium', 'good_similarity': 0.2905, 'failure_similarity': 0.2913, 'confidence': 0.0008}
{'image': 'data/img/img__00009_.png', 'true_label': 'plate', 'quality_score': 0.5008, 'quality_category': 'medium', 'good_similarity': 0.2672, 'failure_similarity': 0.2642, 'confidence': 0.003}
{'image': 'data/img/img__00009_.png', 'true_label': 'plate_dirt', 'quality_score': 0.5043, 'quality_category': 'medium', 'good_similarity': 0.2994, 'failure_similarity': 0.2822, 'confidence': 0.0173}
{'image': 'data/img/img__00009_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.4985, 'quality_category': 'medium', 'good_similarity': 0.2925, 'failure_similarity': 0.2986, 'confidence': 0.0061}
  9%|███████▎                                                                         | 9/100 [00:45<06:23,  4.22s/it]{'image': 'data/img/img__00010_.png', 'true_label': 'bowl', 'quality_score': 0.4999, 'quality_category': 'medium', 'good_similarity': 0.2686, 'failure_similarity': 0.2691, 'confidence': 0.0005}
{'image': 'data/img/img__00010_.png', 'true_label': 'plate', 'quality_score': 0.4953, 'quality_category': 'medium', 'good_similarity': 0.2788, 'failure_similarity': 0.2977, 'confidence': 0.0189}
{'image': 'data/img/img__00010_.png', 'true_label': 'napkin', 'quality_score': 0.5033, 'quality_category': 'medium', 'good_similarity': 0.2855, 'failure_similarity': 0.2724, 'confidence': 0.0132}
{'image': 'data/img/img__00010_.png', 'true_label': 'fork', 'quality_score': 0.5014, 'quality_category': 'medium', 'good_similarity': 0.3109, 'failure_similarity': 0.3053, 'confidence': 0.0056}
{'image': 'data/img/img__00010_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00010_.png', 'true_label': 'glass', 'quality_score': 0.4999, 'quality_category': 'medium', 'good_similarity': 0.2627, 'failure_similarity': 0.2629, 'confidence': 0.0002}
{'image': 'data/img/img__00010_.png', 'true_label': 'plate_dirt', 'quality_score': 0.496, 'quality_category': 'medium', 'good_similarity': 0.2527, 'failure_similarity': 0.2688, 'confidence': 0.0162}
{'image': 'data/img/img__00010_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.4988, 'quality_category': 'medium', 'good_similarity': 0.2741, 'failure_similarity': 0.2788, 'confidence': 0.0047}
 10%|████████                                                                        | 10/100 [00:47<05:40,  3.79s/it]{'image': 'data/img/img__00011_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00011_.png', 'true_label': 'plate', 'quality_score': 0.5038, 'quality_category': 'medium', 'good_similarity': 0.2681, 'failure_similarity': 0.2528, 'confidence': 0.0153}
{'image': 'data/img/img__00011_.png', 'true_label': 'glass', 'quality_score': 0.5045, 'quality_category': 'medium', 'good_similarity': 0.2642, 'failure_similarity': 0.246, 'confidence': 0.0182}
{'image': 'data/img/img__00011_.png', 'true_label': 'bowl', 'quality_score': 0.5014, 'quality_category': 'medium', 'good_similarity': 0.2721, 'failure_similarity': 0.2665, 'confidence': 0.0057}
{'image': 'data/img/img__00011_.png', 'true_label': 'fork', 'quality_score': 0.5007, 'quality_category': 'medium', 'good_similarity': 0.3111, 'failure_similarity': 0.3083, 'confidence': 0.0028}
{'image': 'data/img/img__00011_.png', 'true_label': 'napkin', 'quality_score': 0.5009, 'quality_category': 'medium', 'good_similarity': 0.243, 'failure_similarity': 0.2394, 'confidence': 0.0036}
{'image': 'data/img/img__00011_.png', 'true_label': 'plate_dirt', 'quality_score': 0.5025, 'quality_category': 'medium', 'good_similarity': 0.265, 'failure_similarity': 0.2549, 'confidence': 0.0101}
{'image': 'data/img/img__00011_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.5013, 'quality_category': 'medium', 'good_similarity': 0.2842, 'failure_similarity': 0.2789, 'confidence': 0.0053}
 11%|████████▊                                                                       | 11/100 [00:51<05:23,  3.64s/it]{'image': 'data/img/img__00012_.png', 'true_label': 'napkin', 'quality_score': 0.5066, 'quality_category': 'medium', 'good_similarity': 0.251, 'failure_similarity': 0.2246, 'confidence': 0.0265}
{'image': 'data/img/img__00012_.png', 'true_label': 'glass', 'quality_score': 0.5027, 'quality_category': 'medium', 'good_similarity': 0.2538, 'failure_similarity': 0.2428, 'confidence': 0.011}
{'image': 'data/img/img__00012_.png', 'true_label': 'bowl', 'quality_score': 0.5034, 'quality_category': 'medium', 'good_similarity': 0.2445, 'failure_similarity': 0.2309, 'confidence': 0.0136}
{'image': 'data/img/img__00012_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00012_.png', 'true_label': 'fork', 'quality_score': 0.4993, 'quality_category': 'medium', 'good_similarity': 0.26, 'failure_similarity': 0.2627, 'confidence': 0.0028}
{'image': 'data/img/img__00012_.png', 'true_label': 'plate', 'quality_score': 0.4994, 'quality_category': 'medium', 'good_similarity': 0.2658, 'failure_similarity': 0.2681, 'confidence': 0.0023}
{'image': 'data/img/img__00012_.png', 'true_label': 'plate_dirt', 'quality_score': 0.4992, 'quality_category': 'medium', 'good_similarity': 0.2995, 'failure_similarity': 0.3028, 'confidence': 0.0034}
{'image': 'data/img/img__00012_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.5043, 'quality_category': 'medium', 'good_similarity': 0.2431, 'failure_similarity': 0.226, 'confidence': 0.0171}
 12%|█████████▌                                                                      | 12/100 [00:54<05:01,  3.43s/it]{'image': 'data/img/img__00013_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00013_.png', 'true_label': 'plate', 'quality_score': 0.4945, 'quality_category': 'medium', 'good_similarity': 0.2715, 'failure_similarity': 0.2934, 'confidence': 0.0218}
{'image': 'data/img/img__00013_.png', 'true_label': 'glass', 'quality_score': 0.5041, 'quality_category': 'medium', 'good_similarity': 0.2784, 'failure_similarity': 0.2621, 'confidence': 0.0163}
{'image': 'data/img/img__00013_.png', 'true_label': 'fork', 'quality_score': 0.4996, 'quality_category': 'medium', 'good_similarity': 0.2316, 'failure_similarity': 0.2333, 'confidence': 0.0017}
{'image': 'data/img/img__00013_.png', 'true_label': 'bowl', 'quality_score': 0.4999, 'quality_category': 'medium', 'good_similarity': 0.2504, 'failure_similarity': 0.2508, 'confidence': 0.0005}
{'image': 'data/img/img__00013_.png', 'true_label': 'napkin', 'quality_score': 0.5019, 'quality_category': 'medium', 'good_similarity': 0.312, 'failure_similarity': 0.3044, 'confidence': 0.0077}
{'image': 'data/img/img__00013_.png', 'true_label': 'plate_dirt', 'quality_score': 0.4988, 'quality_category': 'medium', 'good_similarity': 0.3023, 'failure_similarity': 0.3069, 'confidence': 0.0046}
{'image': 'data/img/img__00013_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.5012, 'quality_category': 'medium', 'good_similarity': 0.255, 'failure_similarity': 0.2502, 'confidence': 0.0048}
 13%|██████████▍                                                                     | 13/100 [00:58<05:20,  3.68s/it]{'image': 'data/img/img__00014_.png', 'true_label': 'bowl', 'quality_score': 0.5023, 'quality_category': 'medium', 'good_similarity': 0.2779, 'failure_similarity': 0.2686, 'confidence': 0.0094}
{'image': 'data/img/img__00014_.png', 'true_label': 'napkin', 'quality_score': 0.5016, 'quality_category': 'medium', 'good_similarity': 0.2473, 'failure_similarity': 0.2411, 'confidence': 0.0062}
{'image': 'data/img/img__00014_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00014_.png', 'true_label': 'glass', 'quality_score': 0.5031, 'quality_category': 'medium', 'good_similarity': 0.2387, 'failure_similarity': 0.2261, 'confidence': 0.0126}
{'image': 'data/img/img__00014_.png', 'true_label': 'fork', 'quality_score': 0.4992, 'quality_category': 'medium', 'good_similarity': 0.314, 'failure_similarity': 0.3173, 'confidence': 0.0033}
{'image': 'data/img/img__00014_.png', 'true_label': 'plate', 'quality_score': 0.4974, 'quality_category': 'medium', 'good_similarity': 0.2494, 'failure_similarity': 0.2598, 'confidence': 0.0104}
{'image': 'data/img/img__00014_.png', 'true_label': 'plate_dirt', 'quality_score': 0.502, 'quality_category': 'medium', 'good_similarity': 0.2646, 'failure_similarity': 0.2565, 'confidence': 0.0081}
{'image': 'data/img/img__00014_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.5045, 'quality_category': 'medium', 'good_similarity': 0.2812, 'failure_similarity': 0.2633, 'confidence': 0.0179}
 14%|███████████▏                                                                    | 14/100 [01:00<04:40,  3.26s/it]{'image': 'data/img/img__00015_.png', 'true_label': 'plate', 'quality_score': 0.5005, 'quality_category': 'medium', 'good_similarity': 0.2588, 'failure_similarity': 0.257, 'confidence': 0.0019}
{'image': 'data/img/img__00015_.png', 'true_label': 'bowl', 'quality_score': 0.5024, 'quality_category': 'medium', 'good_similarity': 0.2846, 'failure_similarity': 0.2751, 'confidence': 0.0094}
{'image': 'data/img/img__00015_.png', 'true_label': 'knive', 'quality_score': 0.5, 'quality_category': 'medium', 'good_similarity': 0.0, 'failure_similarity': 0.0, 'confidence': 0.0}
{'image': 'data/img/img__00015_.png', 'true_label': 'napkin', 'quality_score': 0.5039, 'quality_category': 'medium', 'good_similarity': 0.3001, 'failure_similarity': 0.2846, 'confidence': 0.0155}
{'image': 'data/img/img__00015_.png', 'true_label': 'fork', 'quality_score': 0.501, 'quality_category': 'medium', 'good_similarity': 0.3099, 'failure_similarity': 0.306, 'confidence': 0.0038}
{'image': 'data/img/img__00015_.png', 'true_label': 'glass', 'quality_score': 0.4982, 'quality_category': 'medium', 'good_similarity': 0.2979, 'failure_similarity': 0.305, 'confidence': 0.0072}
{'image': 'data/img/img__00015_.png', 'true_label': 'plate_dirt', 'quality_score': 0.5007, 'quality_category': 'medium', 'good_similarity': 0.2729, 'failure_similarity': 0.27, 'confidence': 0.0029}
{'image': 'data/img/img__00015_.png', 'true_label': 'bowl_dirt', 'quality_score': 0.5025, 'quality_category': 'medium', 'good_similarity': 0.2841, 'failure_similarity': 0.2741, 'confidence': 0.01}
´´´
