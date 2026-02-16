from Bio import SeqIO


standard_aa = set("ACDEFGHIKLMNPQRSTVWY")  # standard 20 amino acids
standard_count = 0
non_standard_count = 0

# Count the number of standard and non-standard sequences in the FASTA file
for record in SeqIO.parse("data/uniprot_sprot.fasta", "fasta"):
    if not set(record.seq).issubset(standard_aa):
        print(f"Non-standard sequence found in record {record.id}")
        print(f"Sequence: {record.seq}")
        print(f"Length: {len(record)}")
        print(f"Description: {record.description}")
        non_standard_count += 1
    else:
        print(f"Standard sequence found in record {record.id}")
        print(f"Sequence: {record.seq}")
        print(f"Length: {len(record)}")
        print(f"Description: {record.description}")
        standard_count += 1
print(f"Total standard sequences: {standard_count}")
print(f"Total non-standard sequences: {non_standard_count}")
