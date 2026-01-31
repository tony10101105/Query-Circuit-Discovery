BASE=s3://neuronpedia-datasets/v1/gpt2-small
OUT=sae_data/gpt2-small/32k

for layer in {0..11}; do
  for comp in att mlp; do
    echo "Downloading ${layer}-${comp}_32k-oai"
    aws s3 sync \
      ${BASE}/${layer}-${comp}_32k-oai/ \
      ${OUT}/${layer}-${comp}_32k-oai/ \
      --no-sign-request
  done
done
