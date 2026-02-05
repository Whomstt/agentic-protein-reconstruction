from Bio import SeqIO

for record in SeqIO.parse("data/uniprot_sprot.fasta", "fasta"):
    print(f"ID: {record.id}")
    print(f"Sequence: {record.seq}")
    print(f"Length: {len(record)}")
    print(f"Description: {record.description}")
