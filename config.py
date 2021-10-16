import argparse

class Config:
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--data_path", default="data/MultiWOZ_2.1", type=str)
    parser.add_argument("--db_path", default="data/MultiWOZ_2.1", type=str)
    parser.add_argument("--assets_path", default="assets", type=str)
    parser.add_argument("--save_path", default=None, type=str)

    # modeling
    parser.add_argument("--max_context_len", default=512, type=int)
    parser.add_argument("--max_sentence_len", default=100, type=int)
    parser.add_argument("--max_act_len", default=50, type=int)
    parser.add_argument("--max_value_len", default=10, type=int)
    parser.add_argument("--max_belief_len", default=500, type=int)
    parser.add_argument("--embedding_size", default=768, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--attention_projection_layer", action="store_true", default=False)
    parser.add_argument("--weight_tying", action="store_true", default=False)

    # training
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--min_epochs", default=20, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--early_stop_count", default=5, type=int)
    parser.add_argument("--dynamic_tqdm", action="store_true", default=False)
    parser.add_argument("--gradient_clipping", default=10.0, type=float)
    parser.add_argument("--cuda_device", default=0, type=int)
    
    # testing
    parser.add_argument("--postprocessing", action="store_true", default=False)
    parser.add_argument("--make_report", action="store_true", default=False)

    # vocabulary
    parser.add_argument("--pad_idx", default=0, type=int)
    parser.add_argument("--slot_idx", default=1, type=int)
    parser.add_argument("--eos_idx", default=3, type=int)
    parser.add_argument("--domain_idx", default=4, type=int)
    parser.add_argument("--action_idx", default=8, type=int)
    parser.add_argument("--unk_idx", default=100, type=int)
    parser.add_argument("--cls_idx", default=101, type=int)
    parser.add_argument("--sep_idx", default=102, type=int)
    parser.add_argument("--vocab_size", default=30522, type=int)

    # belief tracking
    parser.add_argument("--num_domains", default=7, type=int)
    parser.add_argument("--num_gates", default=4, type=int)
    parser.add_argument("--delete_idx", default=0, type=int)
    parser.add_argument("--update_idx", default=1, type=int)
    parser.add_argument("--dontcare_idx", default=2, type=int)
    parser.add_argument("--copy_idx", default=3, type=int)

    # for RL
    parser.add_argument("--discount_factor", default=0.99, type=float)
    parser.add_argument("--use_action_rate", action="store_true", default=False)
    parser.add_argument("--negative_action_reward", action="store_true", default=False)
    parser.add_argument("--beta", default=0.001, type=float)
    parser.add_argument("--weighted_action_reward", action="store_true", default=False)

    # for distributed training
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--backend", default="nccl", type=str)
    parser.add_argument("--master_port", default=29500, type=int)