python -m test.tree_test.main --embedding_type='poincare_maps' --output_space='poincare'
python -m test.tree_test.main --embedding_type='poincare_maps' --output_space='spherical'

python -m test.tree_test.link_prediction --embedding_type="poincare_maps" --q=0.9 --n_links=20