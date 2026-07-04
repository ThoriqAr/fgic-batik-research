def select_samples(
    samples,
    evaluation_config
):
  
    if evaluation_config.selected_classes:

      samples = [

          sample

          for sample in samples

          if sample.ground_truth
          in evaluation_config.selected_classes
      ]

    if evaluation_config.samples_per_class:

      grouped = {}

      for sample in samples:

          grouped.setdefault(
              sample.ground_truth,
              []
          ).append(sample)

      selected = []

      for class_name, class_samples in grouped.items():

          selected.extend(
              class_samples[
                  :evaluation_config.samples_per_class
              ]
          )

      samples = selected

    if evaluation_config.selected_attribution_ids:

      samples = [

          sample

          for sample in samples

          if sample.attribution_id
          in evaluation_config.selected_attribution_ids
      ]

    if evaluation_config.max_samples:

      samples = samples[
          :evaluation_config.max_samples
      ]

    return samples