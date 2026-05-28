# Download TCGA data via TCGAbiolinks
if (!requireNamespace("TCGAbiolinks", quietly = TRUE)) {
  install.packages("BiocManager"); BiocManager::install("TCGAbiolinks")
}
library(TCGAbiolinks); library(SummarizedExperiment)
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

save_mat <- function(se, out) write.csv(assay(se), out, quote = FALSE)

get_mrna <- function(project, workflow="HTSeq - FPKM"){
  q <- GDCquery(project=project, data.category="Transcriptome Profiling",
                data.type="Gene Expression Quantification", workflow.type=workflow)
  GDCdownload(q); se <- GDCprepare(q); save_mat(se, file.path("data/raw", paste0(project,"_mRNA_FPKM.csv")))
}
get_mirna <- function(project){
  q <- GDCquery(project=project, data.category="Transcriptome Profiling",
                data.type="miRNA Expression Quantification")
  GDCdownload(q); se <- GDCprepare(q); save_mat(se, file.path("data/raw", paste0(project,"_miRNA.csv")))
}
get_meth <- function(project){
  q <- GDCquery(project=project, data.category="DNA Methylation",
                platform="Illumina Human Methylation 450")
  GDCdownload(q); se <- GDCprepare(q); save_mat(se, file.path("data/raw", paste0(project,"_Meth450.csv")))
}
get_all <- function(project){ get_mrna(project); get_mirna(project); get_meth(project); message("Done: ", project) }
# Example in R: source("scripts/get_tcga_data.R"); get_all("TCGA-BRCA")
