import os

import torch.utils.data
from torchvision import transforms
import numpy as np

class CIFAR10Dataset(torch.utils.data.Dataset):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    num_classes = 10

    def __init__(self, file_path, domains, transform='src', distribution='real', dir_beta=0.1, shuffle_criterion='class'):
        super(CIFAR10Dataset, self).__init__()
        if distribution not in ['real', 'random', 'dirichlet']:
            raise NotImplementedError
        if shuffle_criterion not in ['class', 'domain']:
            raise ValueError('Unknown criterion {}'.format(shuffle_criterion))

        self.file_path = file_path
        self.domains = domains
        self.num_domains = len(domains)
        self.distribution = distribution
        self.dir_beta = dir_beta
        self.shuffle_criterion = shuffle_criterion

        self.features = []
        self.class_labels = []
        self.domain_labels = []

        for i, domain in enumerate(self.domains):
            data_path, label_path = self.get_filepaths(domain)
            data = np.load(data_path)
            data = data.astype('float32')  / 255.0
            data = data.transpose((0, 3, 1, 2)) # To (B, C, H, W)

            self.features.append(torch.from_numpy(data))
            self.class_labels.append(torch.from_numpy(np.load(label_path)).long())
            self.domain_labels.append(torch.Tensor([i for _ in range(len(data))]).long())

        self.features = torch.cat(self.features)
        self.class_labels = torch.cat(self.class_labels)
        self.domain_labels = torch.cat(self.domain_labels)

        # Don't shuffle if source training
        if transform != 'src':
            self.shuffle_distribution()

        self.dataset = torch.utils.data.TensorDataset(
            self.features,
            self.class_labels,
            self.domain_labels,
        )

        if transform == 'src':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        elif transform == 'val':
            self.transform = transforms.Compose([
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, cl, dl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, dl

    def get_filepaths(self, domain):
        if domain.startswith('original'):
            sub_path1 = 'origin'
            sub_path2 = ''
            data_filename = 'original.npy'
            label_filename = 'labels.npy'
        elif domain.startswith('test'):
            sub_path1 = 'corrupted'
            sub_path2 = 'severity-1'
            data_filename = 'test.npy'
            label_filename = 'labels.npy'
        else:
            sub_path1 = 'corrupted'
            sub_path2 = 'severity-' + domain.split('-')[1]
            # sub_path2 = 'severity-1'
            data_filename = domain.split('-')[0] + '.npy'
            label_filename = 'labels.npy'

        data_path = os.path.join(self.file_path, sub_path1, sub_path2, data_filename)
        label_path = os.path.join(self.file_path, sub_path1, sub_path2, label_filename)
        return data_path, label_path

    def shuffle_distribution(self):
        # Originally there's a config variable n_samples, skipped here
        if self.distribution == 'real':
            return

        rng = np.random.default_rng()
        permutation = rng.permutation(len(self.class_labels))

        self.features = self.features[permutation]
        self.class_labels = self.class_labels[permutation]
        self.domain_labels = self.domain_labels[permutation]

        if self.distribution == 'dirichlet':
            numchunks = self.num_classes if self.shuffle_criterion == 'class' else self.num_domains
            min_size = -1
            N = len(self.features)
            min_size_thresh = 10

            result_feats = []
            result_cl_labels = []
            result_do_labels = []

            while min_size < min_size_thresh:
                    idx_batch = [[] for _ in range(numchunks)]
                    idx_batch_cls = [[] for _ in range(numchunks)]  # contains data per each class

                    if self.shuffle_criterion == 'class':
                        criterion = self.class_labels.numpy()
                        num_categories = self.num_classes
                    else:
                        criterion = self.domain_labels.numpy()
                        num_categories = self.num_domains

                    for k in range(num_categories):
                        idx_k = np.where(criterion == k)[0]
                        np.random.shuffle(idx_k)

                        proportions = rng.dirichlet(np.repeat(self.dir_beta, numchunks))

                        # balance
                        proportions = np.array([p * (len(idx_j) < N / numchunks)
                                                for p, idx_j in zip(proportions, idx_batch)])
                        proportions = proportions / proportions.sum()
                        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                        splits = np.split(idx_k, proportions)
                        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, splits)]

                        # store class/domain-wise data
                        for idx_j, idx in zip(idx_batch_cls, splits):
                            idx_j.append(idx)

                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    if min_size >= min_size_thresh:
                        # create temporally correlated dataset by shuffling categories
                        for chunk in idx_batch_cls:
                            categories_seq = list(range(num_categories))
                            np.random.shuffle(categories_seq)
                            for cat in categories_seq:
                                if cat < len(chunk):  # safety check
                                    idx = chunk[cat]
                                    result_feats.extend([self.features[i] for i in idx])
                                    result_cl_labels.extend([self.class_labels[i] for i in idx])
                                    result_do_labels.extend([self.domain_labels[i] for i in idx])

            # Convert lists back to tensors
            self.features = torch.stack(result_feats)
            self.class_labels = torch.LongTensor(result_cl_labels)
            self.domain_labels = torch.LongTensor(result_do_labels)

    def analyze_chunk_distribution(self):
        """
        Analyzes and prints the distribution of labels within chunks after Dirichlet shuffling.
        Returns a dictionary with distribution statistics.
        """
        if self.distribution != 'dirichlet':
            print("This analysis is only meaningful for Dirichlet distribution")
            return

        # Determine chunk size (approximate)
        total_samples = len(self.features)
        numchunks = self.num_classes if self.shuffle_criterion == 'class' else self.num_domains
        chunk_size = total_samples // numchunks
        
        # Initialize counters for each chunk
        chunk_distributions = []
        
        # Analyze each chunk
        for i in range(numchunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < numchunks - 1 else total_samples
            
            if self.shuffle_criterion == 'class':
                labels = self.class_labels[start_idx:end_idx].numpy()
                num_categories = self.num_classes
                category_name = 'Class'
            else:
                labels = self.domain_labels[start_idx:end_idx].numpy()
                num_categories = self.num_domains
                category_name = 'Domain'
            
            # Count occurrences of each label in this chunk
            unique_labels, counts = np.unique(labels, return_counts=True)
            distribution = np.zeros(num_categories)
            distribution[unique_labels] = counts
            
            # Calculate percentages
            percentages = (distribution / len(labels)) * 100
            chunk_distributions.append(percentages)
            
            # Print detailed information for this chunk
            print(f"\nChunk {i + 1} Distribution:")
            print(f"Total samples in chunk: {len(labels)}")
            print(f"{category_name} distribution (%):")
            for label, percentage in enumerate(percentages):
                print(f"{category_name} {label}: {percentage:.2f}%")
                
        # Calculate some statistical measures
        chunk_distributions = np.array(chunk_distributions)
        mean_distribution = np.mean(chunk_distributions, axis=0)
        std_distribution = np.std(chunk_distributions, axis=0)
        
        print("\nOverall Statistics:")
        print(f"Mean distribution across chunks (%):")
        for label, (mean, std) in enumerate(zip(mean_distribution, std_distribution)):
            print(f"{category_name} {label}: {mean:.2f}% ± {std:.2f}")
            
        # Calculate imbalance metrics
        gini_coefficients = [self._calculate_gini(dist) for dist in chunk_distributions]
        print(f"\nGini Coefficients (0 = perfect equality, 1 = perfect inequality):")
        for i, gini in enumerate(gini_coefficients):
            print(f"Chunk {i + 1}: {gini:.3f}")
        print(f"Mean Gini Coefficient: {np.mean(gini_coefficients):.3f}")
        
        return {
            'chunk_distributions': chunk_distributions,
            'mean_distribution': mean_distribution,
            'std_distribution': std_distribution,
            'gini_coefficients': gini_coefficients
        }

    def _calculate_gini(self, array):
        """
        Calculate the Gini coefficient for a distribution array.
        0 represents perfect equality, 1 represents perfect inequality.
        """
        array = np.array(array)
        if np.amin(array) < 0:
            array -= np.amin(array)
        array += 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


if __name__ == '__main__':
    corruptions = ["shot_noise-5", "motion_blur-5", "snow-5", "pixelate-5", "gaussian_noise-5"]
    
    # Create dataset with Dirichlet distribution
    data = CIFAR10Dataset(
        file_path='./dataset/CIFAR-10-C',
        domains=[corruptions[0]],
        transform='val',
        distribution="dirichlet",
        dir_beta=10,
        shuffle_criterion="class"  # or "domain"
    )
    
    # Analyze the distribution
    stats = data.analyze_chunk_distribution()

    print(stats["chunk_distributions"])
    
    # Optional: If you want to visualize the distributions using matplotlib
    # import matplotlib.pyplot as plt
    #
    # def plot_chunk_distributions(stats, criterion="class"):
    #     chunk_distributions = stats['chunk_distributions']
    #     num_chunks = len(chunk_distributions)
    #     num_categories = len(chunk_distributions[0])
    #
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    #
    #     # Plot distribution for each chunk
    #     for i in range(num_chunks):
    #         ax1.bar(np.arange(num_categories) + i * 0.1,
    #                chunk_distributions[i],
    #                width=0.1,
    #                alpha=0.5,
    #                label=f'Chunk {i+1}')
    #
    #     ax1.set_xlabel(f'{"Class" if criterion=="class" else "Domain"} ID')
    #     ax1.set_ylabel('Percentage (%)')
    #     ax1.set_title('Distribution per Chunk')
    #     ax1.legend()
    #
    #     # Plot mean and std
    #     ax2.bar(np.arange(num_categories),
    #             stats['mean_distribution'],
    #             yerr=stats['std_distribution'],
    #             capsize=5)
    #     ax2.set_xlabel(f'{"Class" if criterion=="class" else "Domain"} ID')
    #     ax2.set_ylabel('Mean Percentage (%)')
    #     ax2.set_title('Mean Distribution Across Chunks (with std dev)')
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    # plot_chunk_distributions(stats, data.shuffle_criterion)