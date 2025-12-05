import pandas as pd

real = pd.read_csv('fake_job_postings_og.csv')
fake = pd.read_csv('Only Fake Postings.csv')

fake = fake[['title', 'description', 'requirements', 'company_profile', 'location', 'salary_range', 'benefits', 'fraudulent']]
real = real[['title', 'description', 'requirements', 'company_profile', 'location', 'salary_range', 'benefits', 'fraudulent']]

combined = pd.concat([real, fake], ignore_index=True)
combined = combined.sample(frac=1).reset_index(drop=True)  # Shuffle the combined dataset

combined.to_csv('combined_job_postings.csv', index=False)