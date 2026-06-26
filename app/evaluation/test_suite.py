import json
from typing import List, Dict, Any
from app.core.logging import logger
from pathlib import Path

class TestSuiteGenerator:
    """Generate test suite for evaluation."""
    
    def __init__(self):
        self.test_suite = []
    
    def generate_sample_questions(self) -> List[Dict[str, Any]]:
        """Generate 25 test questions covering various scenarios."""
        test_cases = []
        
        # Bengali direct questions
        bengali_questions = [
            {
                "question": "গল্পের প্রধান নারী চরিত্রের নাম কী?",
                "expected_answer": "গল্পের প্রধান নারী চরিত্রের নাম চিত্রা।",
                "type": "direct_fact",
                "language": "bn",
                "relevant_docs": ["chunk_0006", "chunk_0007"]
            },
            {
                "question": "চিত্রা কোন বিশ্ববিদ্যালয়ে পড়ে?",
                "expected_answer": "চিত্রা ঢাকা বিশ্ববিদ্যালয়ে পড়ে।",
                "type": "direct_fact",
                "language": "bn",
                "relevant_docs": ["chunk_0007"]
            },
            {
                "question": "বৃদ্ধ লোকটির নাম কী?",
                "expected_answer": "বৃদ্ধ লোকটির নাম রশীদ উদ্দিন।",
                "type": "direct_fact",
                "language": "bn",
                "relevant_docs": ["chunk_0007"]
            },
            {
                "question": "আশহাবের মায়ের নাম কী?",
                "expected_answer": "আশহাবের মায়ের নাম সাজেদা বেগম।",
                "type": "direct_fact",
                "language": "bn",
                "relevant_docs": ["chunk_0014", "chunk_0016"]
            },
            {
                "question": "চিত্রা বন্ধুর নাম কী যে তাকে ফোন করে?",
                "expected_answer": "চিত্রা বন্ধুর নাম লিলি।",
                "type": "direct_fact",
                "language": "bn",
                "relevant_docs": ["chunk_0008", "chunk_0009"]
            }
        ]
        
        # English questions
        english_questions = [
            {
                "question": "What is the name of the main female character?",
                "expected_answer": "The main female character's name is Chitra.",
                "type": "direct_fact",
                "language": "en",
                "relevant_docs": ["chunk_0006", "chunk_0007"]
            },
            {
                "question": "What subject does Chitra study?",
                "expected_answer": "Chitra studies Physics.",
                "type": "direct_fact",
                "language": "en",
                "relevant_docs": ["chunk_0007"]
            },
            {
                "question": "What is the name of the old man?",
                "expected_answer": "The old man's name is Rashid Uddin.",
                "type": "direct_fact",
                "language": "en",
                "relevant_docs": ["chunk_0007"]
            },
            {
                "question": "Who is Sajeda Begum?",
                "expected_answer": "Sajeda Begum is Ashab's mother.",
                "type": "direct_fact",
                "language": "en",
                "relevant_docs": ["chunk_0014", "chunk_0016"]
            },
            {
                "question": "What does Ashab do for a living?",
                "expected_answer": "Ashab is a doctor.",
                "type": "direct_fact",
                "language": "en",
                "relevant_docs": ["chunk_0014", "chunk_0066"]
            }
        ]
        
        # Follow-up questions
        followup_questions = [
            {
                "question": "রশীদ উদ্দিন কীভাবে নিজেকে পরিচয় দেন?",
                "expected_answer": "রশীদ উদ্দিন নিজেকে একজন গণিতবিদ হিসেবে পরিচয় দেন এবং নিজেকে 'দৈত্য' (Giant) বলেন। তিনি গণিত জগতের দশজন দৈত্যের একজন।",
                "type": "followup",
                "language": "bn",
                "context": ["রশীদ উদ্দিন ট্রেনের একজন যাত্রী।"],
                "relevant_docs": ["chunk_0026"]
            },
            {
                "question": "What did Rashid Uddin request from Abul Khayer Khan?",
                "expected_answer": "Rashid Uddin requested Abul Khayer Khan to arrange a helicopter to take the woman in labor to a hospital.",
                "type": "followup",
                "language": "en",
                "context": ["A woman was in labor on the train."],
                "relevant_docs": ["chunk_0077", "chunk_0078"]
            },
            {
                "question": "লিলি চিত্রার প্রতি কেন বিরক্তিকর আচরণ করত?",
                "expected_answer": "লিলি চিত্রার প্রতি বিরক্তিকর আচরণ করত কারণ সে হয়ত চিত্রার বন্ধু ছিল এবং তার সাথে ঘনিষ্ঠ সম্পর্কের কারণে মজা করতে চাইত, কিন্তু চিত্রা তার অশ্লীল কথায় বিরক্ত বোধ করত।",
                "type": "followup",
                "language": "bn",
                "context": ["লিলি চিত্রার বন্ধু।"],
                "relevant_docs": ["chunk_0008", "chunk_0009"]
            },
            {
                "question": "Why did Sajeda Begum insist on calling Chitra her daughter-in-law?",
                "expected_answer": "Sajeda Begum insisted on calling Chitra her daughter-in-law because she wanted to see her son Ashab married and she liked Chitha. She saw it as a way to pressure her son into marriage.",
                "type": "followup",
                "language": "en",
                "context": ["Sajeda Begum met Chitra on the train."],
                "relevant_docs": ["chunk_0044", "chunk_0055"]
            }
        ]
        
        # Reasoning questions
        reasoning_questions = [
            {
                "question": "কেন রশীদ উদ্দিন নিজেকে 'দৈত্য' বলে পরিচয় দেন?",
                "expected_answer": "রশীদ উদ্দিন নিজেকে 'দৈত্য' বলে পরিচয় দেন কারণ তিনি গণিত জগতের একজন বিখ্যাত ব্যক্তি এবং তার বই 'Ten Giant of Math World'-এ তার নাম অন্তর্ভুক্ত ছিল।",
                "type": "reasoning",
                "language": "bn",
                "relevant_docs": ["chunk_0026"]
            },
            {
                "question": "What does the train journey symbolize in the story?",
                "expected_answer": "The train journey serves as a microcosm of Bangladeshi society, bringing together different social classes and personalities. It allows various characters to interact and reveals their true natures through crisis situations.",
                "type": "reasoning",
                "language": "en",
                "relevant_docs": ["chunk_0006", "chunk_0007", "chunk_0020"]
            },
            {
                "question": "সাজেদা বেগমের চরিত্রটি কেমন?",
                "expected_answer": "সাজেদা বেগম একজন প্রভাবশালী এবং নিয়ন্ত্রণকারী মা। তিনি তার ছেলের জীবন নিয়ন্ত্রণ করতে চান এবং চিত্রাকে তার পছন্দ হওয়ায় তাকে বৌমা বলে ডাকেন। তিনি তার মতামত স্পষ্টভাবে বলেন।",
                "type": "reasoning",
                "language": "bn",
                "relevant_docs": ["chunk_0014", "chunk_0016", "chunk_0055"]
            },
            {
                "question": "How does the story portray political power in Bangladesh?",
                "expected_answer": "The story suggests that political power and social status are important in Bangladesh. When Abul Khayer Khan loses his ministry, people's attitudes towards him change dramatically. He goes from being powerful to being powerless, and people mock him.",
                "type": "reasoning",
                "language": "en",
                "relevant_docs": ["chunk_0040", "chunk_0050", "chunk_0068"]
            }
        ]
        
        # Additional direct questions from the story
        more_direct_questions = [
            {
                "question": "আশহাবের সাথে চিত্রার পরিচয় কীভাবে হয়?",
                "expected_answer": "আশহাবের সাথে চিত্রার পরিচয় হয় বুফে কারে, যখন আশহাব চিত্রার কাছে বসার অনুমতি চান।",
                "type": "direct_fact",
                "language": "bn",
                "relevant_docs": ["chunk_0012", "chunk_0020"]
            },
            {
                "question": "What happened to the woman in labor?",
                "expected_answer": "The woman gave birth to a baby boy with Ashab's help. The baby was born healthy despite the difficult circumstances.",
                "type": "direct_fact",
                "language": "en",
                "relevant_docs": ["chunk_0066", "chunk_0070"]
            },
            {
                "question": "আবুল খায়ের খান কে ছিলেন?",
                "expected_answer": "আবুল খায়ের খান ছিলেন একজন মন্ত্রী যিনি তার মন্ত্রিত্ব হারান। তিনি ট্রেনের সেলুন কারে ছিলেন এবং তার স্ত্রী সুরমার সাথে ছিলেন।",
                "type": "direct_fact",
                "language": "bn",
                "relevant_docs": ["chunk_0035", "chunk_0040"]
            },
            {
                "question": "What was Abul Khayer Khan's reaction to losing his ministry?",
                "expected_answer": "Abul Khayer Khan was shocked and worried. He was concerned about how people would treat him now that he no longer had political power.",
                "type": "direct_fact",
                "language": "en",
                "relevant_docs": ["chunk_0040", "chunk_0050"]
            },
            {
                "question": "হেলিকপ্টার আসার ব্যবস্থা করতে আবুল খায়ের খান কেন প্রথমে রাজি হননি?",
                "expected_answer": "আবুল খায়ের খান প্রথমে রাজি হননি কারণ তিনি তখন আর মন্ত্রী ছিলেন না এবং তিনি মনে করতেন তার আর ক্ষমতা নেই।",
                "type": "reasoning",
                "language": "bn",
                "relevant_docs": ["chunk_0077", "chunk_0078"]
            }
        ]
        
        # All test cases - combine to exactly 25
        self.test_suite = (
            bengali_questions[:5] + 
            english_questions[:5] + 
            followup_questions[:4] + 
            reasoning_questions[:4] + 
            more_direct_questions[:7]
        )
        
        # Ensure we have exactly 25 questions
        if len(self.test_suite) > 25:
            self.test_suite = self.test_suite[:25]
        
        # Add context to follow-up questions
        for test in self.test_suite:
            if 'context' not in test:
                test['context'] = []
        
        logger.info(f"Generated {len(self.test_suite)} test questions")
        return self.test_suite
    
    def save_test_suite(self, output_path: str):
        """Save test suite to file."""
        if not self.test_suite:
            self.generate_sample_questions()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_suite, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test suite saved to {output_path}")