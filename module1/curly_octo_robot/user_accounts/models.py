from django.db import models
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser

class CORAccountManager(BaseUserManager):
    def create_user(self, email, username, latitude, longitude, password=None):
        if not email:
            return ValueError("Users must have email address")
        if not username:
            return ValueError("Users must have email address")
        if not latitude or not longitude:
            return ValueError("Users must have specified location")
        user = self.model(
            email = self.normalize_email(email),
            username=username,
            latitude=latitude,
            longitude=longitude
        )
        user.set_password(password)
        user.save(user=self._db)
        return user

    def create_superuser(self, email, username, latitude, longitude, password):
        user = self.model(
            email = self.normalize_email(email),
            username=username,
            latitude=latitude,
            longitude=longitude
        )
        user.is_admin = True
        user.set_password(password)
        user.save(using=self._db)
        return user


class Account(AbstractBaseUser):
    email = models.EmailField(
        verbose_name='email',
        max_length=60,
        unique=True
    )
    username = models.CharField(
        max_length=30,
        unique=True
    )
    street_address = models.TextField(
        max_length=100
    )
    is_admin = models.BooleanField(default=False)
    landmark = models.TextField(max_length=100)
    city = models.CharField(max_length=60)
    state = models.CharField(max_length=60)
    pincode = models.CharField(max_length=10)
    latitude = models.FloatField()
    longitude = models.FloatField()
    is_staff = models.BooleanField(default=True)


    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'latitude', 'longitude']

    objects = CORAccountManager()
    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        return self.is_admin
    
    def has_module_perms(self, app_label):
        return True