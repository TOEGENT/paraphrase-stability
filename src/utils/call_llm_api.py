import os
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_URL")
MODEL = os.getenv("MODEL")
def call_llm_api(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5,
        "logprobs": True, 
        "top_logprobs": 1,    
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    response = requests.post(BASE_URL, json=payload, headers=headers,timeout=(10,360))
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    print(call_llm_api("""
You are an emotion classifier. Analyze the text and return ONLY the label name.

Here are some examples:
Text: 'I had the lucky chance to score a Living Social voucher to attend CrossFit Las Vegas's On Ramp introductory session.  It is definitely introductory, but the WODs are still pretty intense and you get a pretty good work out.  My OnRamp buddies were pretty lucky because our group got Joe Marsh as our instructor. He definitely knows his CrossFit!  He was able to teach us all of the right form and techniques for the majority of the CrossFit exercises as well as any progressions if we could not physically do the exercise yet.  He even taught us different ways of stretching to help the tense muscles you get from CrossFit work outs.  He likes to quiz the class to ensure that everyone knows the answers and he has a way of getting everyone in the class involved and pumped to get a goodwork out.\n\nI definitely enjoyed the OnRamp session and plan to join CrossFit Las Vegas on my next paycheck.  CrossFit is not your normal workout regimen, it's definitely going to be a change of lifestyle and I can't wait to begin that journey with CrossFit Las Vegas!'
Label: Positive

Text: 'Went for breakfast to try for the first time. Ordered the baked French toast. Eggs were over cooked, bacon was cold and French toast was more like a bread pudding.'
Label: Negative

Text: 'Music, music, music - oh my!\n\nTao Beach is good for drinking, lounging, socializing, and people-watching to Top 40/dance music. \n\nThe lounge chairsfill up fast so you'll want to come early or reserve a spot. The pool itself is pretty small. Towards the side of the bar, there is also a jacuzzi which is about roughly the same size.\n\n3 stars for the bar/pool/jacuzzi. 1 more star for the music.'
Label: Positive

Text: 'After receiving coupons for so long, we finally decided to try this place. DO NOT waste your money!!!!  It is the WORST place I have ever eaten!!!  Nasty!!!  No flavor, small bits of fatty boiled chicken and way over cooked rice.  I mean it's SO BAD that I wouldn't even have my dogs eat it!!!  Because the place was empty I'm sure the food was not fresh.  I picked up my orders.  I called to inquire why it was so bad but the guy would not speak to me even I was very calm and polite.  Nobody would call me back.  I've seen the other reviews on bad food & lousy customer service.  No wonder they give out so many coupons.  Saveyour money & go to PF Changs for the same price and get quality!!!  These people are a rip off & don't deserve to be in business!!'
Label: Negative


Now analyze this text:
Text: 'This is a great modern iteration of the classic pub... Todd English has brought gourmet to pub and the result is a fun and tasty experience.  The beer selections are fantastic!  Lots of interesting options, now including Napa Smith Organic IPA.  \n\nFood:  Must try the pretzels with cheese, the pot sticker apptzer.  Also sliders and of course fish n chips.  We had table of 5 and ordered a bunch of stuff to share.  Go into this thinking \""pub food with flair\"".\n\nWill definitely be back.'
Label:

"""))