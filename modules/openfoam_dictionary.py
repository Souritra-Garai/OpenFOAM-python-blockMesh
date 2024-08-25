from pathlib import Path

openfoam_version = '9'

file_separator	= '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //'
file_EOF		= '// ************************************************************************* //'
file_header		= '''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  {version}
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
'''.format(version=openfoam_version)

def appendDictionary(file, input_dict: dict, indent:int=0) -> None:

	max_key_length = max([len(str(key)) for key in input_dict.keys()])

	n = len(input_dict)

	for key, value in input_dict.items() :

		if isinstance(value, dict) :

			file.write('\t'*indent + '{key:<{width}}\n'.format(key=key, width=max_key_length) + '\t'*indent + '{\n')

			appendDictionary(file, value, indent+1)

			file.write('\t'*indent + '}\n')

			if n > 1 :
			
				file.write('\n')

		else :

			file.write('\t'*indent + '{key:<{width}}\t{value};\n'.format(key=key, width=max_key_length, value=value))

		n -= 1

	pass

def addHeader(file_path: Path, file_dict:dict) -> None:

	with open(file_path, 'w') as file :

		file.write(file_header)

		file.write('FoamFile\n{\n')

		appendDictionary(file, file_dict, 1)

		file.write('}\n')

		file.write(file_separator)

		file.write('\n')
