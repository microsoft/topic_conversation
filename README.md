# Topic-Conversation Relevance (TCR) Dataset and Benchmarks
---

To help improve meeting effectiveness by understanding if the conversation is on topic, we create a comprehensive Topic-Conversation Relevance (TCR) Dataset that covers a variety of domains and meeting styles. Please refer to the paper _Topic-Conversation Relevance (TCR) Dataset and Benchmarks_ for details.

## Dataset Access
* Data can be downloaded into the ```.\data``` folder [from here](https://mtmdata.blob.core.windows.net/data-share/topic_conversation.zip).
* The TCR dataset includes 1,500 unique meetings, 22,000 words in transcripts, and over 15,000 meeting topics, sourced from both newly collected Speech Interruption Meeting (SIM) data and existing public datasets. 
* Please refer to the paper for input data sources and citations.

## Data Schema
* All data files are in JSON format. They can be downloaded from the link above.
* Data schema overview:

    <img src="img/data_schema.PNG" alt="Data Schema" width="1400">

## Use Data

* The raw data can be downloaded from the link above and should be put under the ```.\data``` folder. They can be parsed based on the schema presentated above.

* Addtional scripts

    * Create synthetic ones from SIM data. [[script](script_create_synthetic_meetings_SIM.py)]
        * The ```SIM_syn100``` data created by this script is included in the ```.\data``` folder.
        * Run script (example):
        
            ```
            python script_create_synthetic_meetings_SIM.py --new_name syn100 --n_syn 100 --rdseed 2
            ```

    * Augment dataset by adding or removing topics. [[script](script_augment_data.py)]
        * The sample data created by this script is included in the ```.\data\example_aug``` folder.
        * Run script (example):

            ```
            python script_augment_data.py --input_data ICSI --new_name aug --variation_addTopics 0.3 --variation_removeTopics 0.3
            ```

## License
Refer to the [LICENSE](LICENSE) file for details.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademark Notice
Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.