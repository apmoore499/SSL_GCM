two extra refs for different algorithm:


Adaptive Semi-Supervised Feature Selection for Cross-Modal Retrieval




</media/krillman/240GB_DATA/codes2/SSL_GCM/reply_and_extra_18_11_2023/Adaptive_Semi-Supervised_Feature_Selection_for_Cross-Modal_Retrieval.pdf>

But, unsupervised methods only considered the pairwise cor-
relations among heterogeneous data and ignored the explicit
high-level semantics in subspace learning

Recently, semi-supervised subspace learning has achieved
promising performance for cross-modal retrieval [34], [35],
since semi-supervised learning can not only use the label infor-
mation but also explore the potential information of unlabeled
data.

And we propose to update
the mapping matrices and the label matrix of unlabeled data
simultaneously and iteratively, so that it can make samples from
different classes to be far apart while those from the same class
lie as close as possible. 

Furthermore, we impose the l2,1 -norm
constraint to select the informative and discriminative features
and reduce the effects of outliers caused by unlabeled data when
learning the mapping matrices.

In details, we use the semantic regression term to explore
the label information, which aims at making the data within the
same class as close as possible in the common latent subspace.

For unlabeled data, we
use the graph model to predict the label matrix as the initial
value.


In other words, we regard the query modal
data as the vertices in graph so that the structure information
of query modal data can be emphasized. Thus, we regard all
labeled and unlabeled data from query modality as the vertices
in the graph, and the edge weights can be formulated as: eq(2)


-------------------------------------------------------------------------

- this method considers problems where we have two distinct features X_1 and X_2 which could be used to infer information about each other, such as in the example of text and images.

- to adapt for our problem context, we consider adapting the datasets D3, D4, D5, D6, D7.

- In D6, we can concatenate X_S and X_E. Alternatively, we can remove X_S and just consider X_C and X_E

- In D7, we can concatenate X_S and X_E. Alternatively, we can remove X_S and just consider X_C and X_E




-------------------------------------------------------------------------

























Semisupervised Feature Analysis by Mining Correlations Among Multiple Tasks

</media/krillman/240GB_DATA/codes2/SSL_GCM/reply_and_extra_18_11_2023/Semisupervised_Feature_Analysis_by_Mining_Correlations_Among_Multiple_Tasks.pdf>



TASKS:


- create synthetic data with 10,000 unlabelled [x]
- create synthetic data with 100,000 unlabelled [x]
- test code with single example from synthetic data to check everything working [x]


- batch slurm for benchmark methods


- partial [x] complete
- fully   [x]
- sslgan  [x]
- sslvae  [x]
- vat     [x]
- triple gan          [x]
- label propagation   [x]



extra approaches
- implement aproach 1: adaptive semisupervised feature selection for cross modal retrieval []
- implement approach 2: semisupervised feature analysis by mining correlations among multiple tasks []


- test new approach 1 []
- test new aproach 2 []


- batch new approach 1: 10,000 unlabelled []
- batch new approach 1: 100,000 unlabelled []

- batch new approach 2: 10,000 unlabelled []
- batch new approach 2: 100,000 unlabelled []


- 