







dns=[i for i in range(1,8)]


methods=[
'ENTROPY_MINIMISATION',
'VAT',
'TRIPLE_GAN',
'SSL_GAN',
'SSVAE',
'LABEL_PROPAGATION']


for METHOD in methods:
	print('')
	for DN in dns:
		template=f"sbatch slurm/{METHOD}_dn_n36_gaussian_mixture_d{DN}_10000.slurm"
		print(template)


