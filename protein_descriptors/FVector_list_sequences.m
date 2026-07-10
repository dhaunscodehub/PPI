function [vectors] = FVector_list_sequences(proteins)
%  F_VECTOR F-vector of protein sequence is constructed in the following manner.
% 
% "FCTP-WSRC: Protein–Protein Interactions Prediction via Weighted Sparse Representation 
% Based Classification"
% 
% doi: 10.3389/fgene.2020.00018
n = length(proteins);
vectors = ones(n,40);
for i = 1:n
    SEQ = proteins(i);
    SEQ = cell2mat(SEQ);
    D = FVector(SEQ);
    vectors(i,:) = D;
end
end