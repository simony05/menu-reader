import boto3

def translate(text):
    translate = boto3.client('translate')
    result = translate.translate_text(
        Text = text,
        SourceLanguageCode = 'zh',
        TargetLanguageCode = 'en'
    )
    return result['TranslatedText']
