import kagglehub

# Download latest version
path = kagglehub.competition_download('the-2nd-ura-hackathon')

print("Path to competition files:", path)