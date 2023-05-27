wheels_folder="wheels"

# Loop through the wheels in the folder
for wheel_file in "$wheels_folder"/*.whl; do
    echo "Installing $wheel_file..."
    /cluster/scratch/$USER/IMC/bin/pip install "$wheel_file" --no-deps
done

/cluster/scratch/$USER/IMC/bin/pip install wheels/hloc-1.3-py3-none-any.whl
