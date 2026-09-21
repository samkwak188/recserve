"""Initial independent account and model schema."""
from alembic import op
from recserve_app.schema_v1 import metadata

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    metadata.create_all(op.get_bind())


def downgrade():
    raise RuntimeError('Destructive production downgrade is not supported; restore or roll forward')
