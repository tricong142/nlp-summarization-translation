import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

class LabelSmoothingLoss(nn.Module):
    """
    Hàm mất mát Cross-Entropy tích hợp Label Smoothing ε=0.1.
    Thay vì ép mô hình dự đoán nhãn đúng với xác suất 100%, 
    nó phân bổ một phần nhỏ xác suất (ε) cho các nhãn sai, 
    giúp mô hình tổng quát hóa tốt hơn trên tập dữ liệu nhỏ.
    """
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # logits: [batch_size * seq_len, vocab_size]
        # labels: [batch_size * seq_len]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Mask bỏ qua các token padding
        non_pad_mask = (labels != self.ignore_index)
        
        # Tính NLL Loss chuẩn
        nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
        
        # Tính Smooth Loss (trung bình trên tất cả các lớp)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Kết hợp
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss[non_pad_mask].mean()

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len, max_target_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        doc = item.get("retrieval_context", item.get("document", ""))
        summ = item.get("summary", "")
        
        inputs = self.tokenizer(doc, max_length=self.max_input_len, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(summ, max_length=self.max_target_len, truncation=True, padding="max_length", return_tensors="pt")
        
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()
        
        # Đặt index = -100 cho các token padding để hàm Loss bỏ qua chúng
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class BARTFinetuner:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

    def _evaluate(self, model, dataloader, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def finetune(self, train_data: list, val_data: list) -> str:
        if not self.config.RUN_FINETUNING:
            print(f"  [BART] Bỏ qua Fine-tuning (RUN_FINETUNING = False). Dùng Zero-shot model: {self.config.BART_MODEL}")
            return self.config.BART_MODEL

        print(f"  [BART] Khởi động Fine-tuning với Label Smoothing={self.config.FINETUNE_LABEL_SMOOTHING} và FP16={self.config.FINETUNE_FP16}...")
        
        from transformers import BartForConditionalGeneration, BartTokenizer, get_linear_schedule_with_warmup
        
        tokenizer = BartTokenizer.from_pretrained(self.config.BART_MODEL)
        model = BartForConditionalGeneration.from_pretrained(self.config.BART_MODEL).to(self.device)
        
        train_dataset = SummarizationDataset(train_data, tokenizer, self.config.FINETUNE_MAX_INPUT_LEN, self.config.FINETUNE_MAX_TARGET_LEN)
        val_dataset = SummarizationDataset(val_data, tokenizer, self.config.FINETUNE_MAX_INPUT_LEN, self.config.FINETUNE_MAX_TARGET_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.FINETUNE_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.FINETUNE_BATCH_SIZE, shuffle=False)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.FINETUNE_LR, weight_decay=0.01)
        
        total_steps = len(train_loader) * self.config.FINETUNE_EPOCHS // self.config.FINETUNE_GRAD_ACCUM
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.FINETUNE_WARMUP_STEPS, num_training_steps=total_steps)
        
        criterion = LabelSmoothingLoss(smoothing=self.config.FINETUNE_LABEL_SMOOTHING)
        scaler = GradScaler() if AMP_AVAILABLE and self.config.FINETUNE_FP16 else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 3 
        
        os.makedirs(self.config.FINETUNED_MODEL_DIR, exist_ok=True)

        for epoch in range(self.config.FINETUNE_EPOCHS):
            model.train()
            train_loss = 0.0
            
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Tính toán trong không gian Mixed Precision (FP16)
                with autocast(enabled=AMP_AVAILABLE and self.config.FINETUNE_FP16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    loss = loss / self.config.FINETUNE_GRAD_ACCUM
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                train_loss += loss.item() * self.config.FINETUNE_GRAD_ACCUM
                
                # Gradient Accumulation
                if (step + 1) % self.config.FINETUNE_GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                    scheduler.step()
                    optimizer.zero_grad()
            
            # Đánh giá sau mỗi Epoch
            val_loss = self._evaluate(model, val_loader, criterion)
            print(f"  [Epoch {epoch+1}/{self.config.FINETUNE_EPOCHS}] Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model.save_pretrained(self.config.FINETUNED_MODEL_DIR)
                tokenizer.save_pretrained(self.config.FINETUNED_MODEL_DIR)
                print(f"    ✓ Đã lưu Checkpoint tốt nhất tại Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"  [BART] Early Stopping được kích hoạt tại epoch {epoch+1} để tránh Overfitting.")
                    break
                    
        return self.config.FINETUNED_MODEL_DIR