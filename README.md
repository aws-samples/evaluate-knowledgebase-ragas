## Evaluate a knowledge base with RAGAS

This repository contains **unofficial samples** to evaluate a [Bedrock knowledge base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) with the [RAGAS framework](https://docs.ragas.io/en/stable/)

1. Run the **prereq_setup.sh** script to install Python and RAGAS

./prereq_setup.sh

2. Run the **ragas_evaluate_bedrock_knowledgebase.sh** script to evaluate your Bedrock knowledge base

python ragas_evaluate_bedrock_knowledgebase.py \
    --region "#region#" \
    --model-id "#model-id#" \
    --embedding-model-id "#embedding-model-id#" \
    --knowledgebase-id "#bedrock-knowledgebase-id#" \
    --evalresult-path "./ragas_evaluationresult.csv" \
    --testplan-path "./ragas_sample_testplan.csv"

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

