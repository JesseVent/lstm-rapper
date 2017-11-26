from flask_wtf import FlaskForm
from wtforms import StringField, HiddenField


class SonnetForm(FlaskForm):
    seed_phrase = StringField('seed_phrase')
    seed = StringField('seed')
    seed_tag = StringField('seed_tag')
