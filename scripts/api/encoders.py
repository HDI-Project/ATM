import datetime
import decimal
import uuid

import simplejson as json
from sqlalchemy import inspect


def object_as_dict(obj):
        return {c.key: getattr(obj, c.key) for c in inspect(
            obj).mapper.column_attrs}  # noqa


class JSONEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that knows how to encode date/time, decimal types, and
    UUIDs.
    See: https://stackoverflow.com/questions/11875770/how-to-overcome-datetime-datetime-not-json-serializable # noqa
    """

    def default(self, o):
        # See "Date Time String Format" in the ECMA-262 specification.
        if isinstance(o, datetime.datetime):
            r = o.isoformat()
            if o.microsecond:
                r = r[:23] + r[26:]
            if r.endswith('+00:00'):
                r = r[:-6] + 'Z'
            return r
        elif isinstance(o, datetime.date):
            return o.isoformat()
        elif isinstance(o, datetime.time):
            if o.utcoffset() is not None:
                raise ValueError("JSON can't represent timezone-aware times.")
            r = o.isoformat()
            if o.microsecond:
                r = r[:12]
            return r
        elif isinstance(o, (decimal.Decimal, uuid.UUID)):
            return str(o)
        else:
            return super(JSONEncoder, self).default(o)


def encode_entity(entity=[]):
    """
    Creates a generic controller function to filter the entity by the value of
    one field.

    Uses simplejson (aliased to json) to parse Decimals and the custom
    JSONEncoder to parse datetime fields.
    """
    return json.dumps([object_as_dict(x) for x in entity], cls=JSONEncoder)
