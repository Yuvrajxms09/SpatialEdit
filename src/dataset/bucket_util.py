class BucketGroup:
    """Manages dynamic batch grouping buckets for mixed video and image training."""

    def __init__(
        self,
        bucket_configs: list[tuple[int, int, int, int, int]],
        prioritize_frame_matching: bool = True,
    ):
        """
        Initialize bucket group with predefined configurations.

        Args:
            bucket_configs: List of (batch_size, num_items, num_frames, height, width) tuples
            prioritize_frame_matching: If True, prioritize frame count matching for videos,
                                     otherwise prioritize aspect ratio matching first
        """
        self.bucket_configs = [tuple(b) for b in bucket_configs]
        self.prioritize_frame_matching = prioritize_frame_matching

    def find_best_bucket(self, media_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
        """
        Find the best matching bucket for given media dimensions.

        Args:
            media_shape: (num_items, num_frames, height, width) of input media

        Returns:
            Best matching bucket as (batch_size, num_items, num_frames, height, width)

        Matching Rules:
            For Images (num_frames=1):
                - Find bucket with num_frames=1 and closest aspect ratio

            For Videos (num_frames>1):
                - If prioritize_frame_matching=True:
                  1. Find buckets with max valid frame count (≤ input frames)
                  2. Among those, select the one with closest aspect ratio
                - If prioritize_frame_matching=False:
                  1. Find buckets with closest aspect ratio
                  2. Among those, select the one with max valid frame count
        """
        num_items, num_frames, height, width = media_shape
        target_aspect_ratio = height / width

        if num_frames == 1:
            valid_buckets = []
            for bucket in self.bucket_configs:
                if bucket[1] == num_items and bucket[2] == 1:
                    valid_buckets.append(bucket)

            if len(valid_buckets) == 0:
                raise ValueError(
                    f"No image buckets found for shape {media_shape}")

            return min(
                valid_buckets,
                key=lambda bucket: abs(
                    (bucket[3] / bucket[4]) - target_aspect_ratio)
            )
        else:
            valid_buckets = []
            for bucket in self.bucket_configs:
                if bucket[1] == num_items and bucket[2] > 1 and bucket[2] <= num_frames:
                    valid_buckets.append(bucket)

            if len(valid_buckets) == 0:
                raise ValueError(
                    f"No video buckets found for shape {media_shape}")

            if self.prioritize_frame_matching:
                max_frame_count = max(bucket[2] for bucket in valid_buckets)
                max_frame_buckets = [
                    bucket for bucket in valid_buckets if bucket[2] == max_frame_count]

                return min(
                    max_frame_buckets,
                    key=lambda bucket: abs(
                        (bucket[3] / bucket[4]) - target_aspect_ratio)
                )
            else:
                min_ratio_difference = min(
                    abs((bucket[3] / bucket[4]) - target_aspect_ratio) for bucket in valid_buckets)
                best_ratio_buckets = [
                    bucket for bucket in valid_buckets
                    if abs((bucket[3] / bucket[4]) - target_aspect_ratio) == min_ratio_difference
                ]

                return max(best_ratio_buckets, key=lambda bucket: bucket[2])

    def __repr__(self) -> str:
        return (
            f"BucketGroup("
            f"total_buckets={len(self.bucket_configs)}, "
            f"prioritize_frame_matching={self.prioritize_frame_matching}, "
            f"configs={self.bucket_configs})"
        )
