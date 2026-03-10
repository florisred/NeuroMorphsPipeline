
import re


class PlotCreatorMixInHelper:



    @staticmethod
    def sort_morphs(data, labels):
        """
        Returns a list of integer indices that sorts the input
        according to the cyclical material path.
        """
        morph_names = labels['morph_name'].values
        anchor_order = sorted(labels[labels['stim_type'] == 'anchor']['morph_name'].values)

        path = []
        for i in range(len(anchor_order)):
            start = anchor_order[i]
            end = anchor_order[(i + 1) % len(anchor_order)]
            path.append((start, end))

        sorted_names = []
        for start, end in path:
            # 1. Add the anchor
            if start in morph_names:
                sorted_names.append(start)

            # 2. Find and sort the transition leg
            leg = []
            for m in morph_names:
                if start in m and end in m and start != end:
                    match = re.search(rf"{start}_([\d.]+)", m)
                    if match:
                        weight = float(match.group(1))
                        leg.append((m, weight))

            leg.sort(key=lambda x: x[1], reverse=True)
            sorted_names.extend([m[0] for m in leg])

        # --- THE INTEGRATION STEP ---
        # Create a mapping of {name: desired_position}
        name_to_pos = {name: i for i, name in enumerate(sorted_names)}

        # Generate the index array based on where each current label should go
        # This handles duplicates if your 'morph_names' has multiple rows per morph
        sorted_idx = sorted(range(len(morph_names)),
                            key=lambda k: name_to_pos.get(morph_names[k], 999))

        sorted_labels = labels.iloc[sorted_idx]
        sorted_data = data[sorted_idx]

        return sorted_data, sorted_labels, sorted_idx
